import collections
from ppo import PPO, _yield_minibatches, _mean_or_nan
import torch
from torch import nn
from torch.nn import functional as F

def loss_fn_kd(outputs, teacher_outputs, T):
    """
    Credit:
    https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py

    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    return nn.KLDivLoss()(
        F.log_softmax(outputs/T, dim=1), 
        F.softmax(teacher_outputs/T, dim=1))

class PPO_Distill(PPO):
    saved_attributes = ("model", "optimizer", "obs_normalizer")

    def __init__(self, *args, **kwargs):
        # EDIT Distill: Reassign model to distill_net for main inference
        self.teacher_model = kwargs['model'] # assigning original model
        kwargs['model'] = kwargs['model'].distill_net
        super().__init__(*args, **kwargs)
        self.model = self.teacher_model.distill_net
        self.kl_record = collections.deque(maxlen=100)

    def _update(self, dataset):
        """Update both the policy and the value function."""

        device = self.device

        if self.obs_normalizer:
            self._update_obs_normalizer(dataset)

        assert "state" in dataset[0]
        assert "v_teacher" in dataset[0]

        if self.standardize_advantages:
            all_advs = torch.tensor([b["adv"] for b in dataset], device=device)
            std_advs, mean_advs = torch.std_mean(all_advs, unbiased=False)

        # Modification: I think OpenAI baselines.ppo2 has a bug here.
        # for batch in _yield_minibatches(
        #         dataset, minibatch_size=self.minibatch_size, num_epochs=self.epochs
        # ):
        batch_size = len(dataset) // self.minibatch_size
        for batch in _yield_minibatches(
            dataset, minibatch_size=batch_size, num_epochs=self.epochs
        ):
            states = self.batch_states(
                [b["state"] for b in batch], self.device, self.phi
            )
            if self.obs_normalizer:
                states = self.obs_normalizer(states, update=False)
            actions = torch.tensor([b["action"] for b in batch], device=device)

            # EDIT Distill: Edited to forward_distill
            # def forward_distill(self, obs, cl_func, kl_func)
            # --> dist, value, dist_dist, value_dist, corr_loss, kl_loss
            kl_func = lambda x, y: loss_fn_kd(x, y, self.model.T)
            cl_func = self.model.cl_func
            alpha= self.model.alpha
            _, _, distribs, vs_pred, corr_loss, kl_loss =\
                self.teacher_model.forward_distill(states, cl_func, kl_func)

            advs = torch.tensor(
                [b["adv"] for b in batch], dtype=torch.float32, device=device
            )
            if self.standardize_advantages:
                advs = (advs - mean_advs) / (std_advs + 1e-8)

            log_probs_old = torch.tensor(
                [b["log_prob"] for b in batch], dtype=torch.float, device=device,
            )
            vs_pred_old = torch.tensor(
                [b["v_pred"] for b in batch], dtype=torch.float, device=device,
            )
            vs_teacher = torch.tensor(
                [b["v_teacher"] for b in batch], dtype=torch.float, device=device,
            )
            # Same shape as vs_pred: (batch_size, 1)
            vs_pred_old = vs_pred_old[..., None]
            vs_teacher = vs_teacher[..., None]

            self.model.zero_grad()
            # EDIT Distill: Uses alpha for distillation weight and 
            # corr_loss (if specified) and kl_loss
            loss = alpha * self._lossfun(
                distribs.entropy(),
                vs_pred,
                distribs.log_prob(actions),
                vs_pred_old=vs_pred_old,
                log_probs_old=log_probs_old,
                advs=advs,
                vs_teacher=vs_teacher,
            ) + (1.0 - alpha) * (corr_loss + kl_loss)
            self.kl_record.append(kl_loss.item())

            loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
            self.optimizer.step()
            self.n_updates += 1

    def get_statistics(self):
        return [
            ("average_value", _mean_or_nan(self.value_record)),
            ("average_entropy", _mean_or_nan(self.entropy_record)),
            ("average_value_loss", _mean_or_nan(self.value_loss_record)),
            ("average_policy_loss", _mean_or_nan(self.policy_loss_record)),
            ("average_kl_loss", _mean_or_nan(self.kl_record)),
            ("n_updates", self.n_updates),
            ("explained_variance", self.explained_variance),
        ]
