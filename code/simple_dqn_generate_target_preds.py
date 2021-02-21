    def _generate_target_preds(self, batch: BatchOfExperiences,
                               preds: torch.Tensor, reward: float) -> torch.Tensor:
        previous_states, previous_actions, \
            previous_possible_actions, next_states = batch
        discounted_reward_sums = \
            self._calculate_discounted_reward_sums(
                len(previous_actions), reward)

        next_preds = self._policy_net(torch.cat(next_states))
        next_qs = next_preds.max(dim=1).values.detach()

        target_preds = torch.zeros(preds.shape).to(self._device)

        for i, ppa in enumerate(previous_possible_actions):
            target_preds[i][list(ppa)] = preds[i][list(ppa)]
            if i == len(previous_possible_actions) - 1:
                target_preds[i][previous_actions[i]] = \
                    discounted_reward_sums[i]
            else:
                target_preds[i][previous_actions[i]] = \
                    (next_qs[i] * self._gamma) + discounted_reward_sums[i]

        return target_preds
