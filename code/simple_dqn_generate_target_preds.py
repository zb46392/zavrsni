    def _generate_target_preds(self, batch: Batch,
                               preds: torch.Tensor,
                               reward: int) -> torch.Tensor:
        previous_moves = batch[1]
        previous_possible_actions = batch[2]
        next_states = batch[3]
        discounted_reward_sums = \
            self._calculate_discounted_reward_sums(reward, len(previous_moves))

        next_preds = self._policy_net(torch.cat(next_states))
        next_qs = next_preds.max(dim=1).values.detach()

        target_preds = torch.zeros(preds.shape).to(self._device)

        for i, ppa in enumerate(previous_possible_actions):
            indices = [self._actions_indices[a] for a in ppa]
            previous_action_i = self._actions_indices.get(previous_moves[i])
            target_preds[i][indices] = preds[i][indices]
            if ppa == previous_possible_actions[-1]:
                target_preds[i][previous_action_i] = discounted_reward_sums[i]
            else:
                target_preds[i][previous_action_i] = (next_qs[i] * self._gamma) \
                                                     + discounted_reward_sums[i]

        return target_preds
