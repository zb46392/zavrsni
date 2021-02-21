    def _update_policy_net(self, batch: BatchOfExperiences,
                           reward: float) -> None:
        previous_states, previous_actions, \
            previous_possible_actions, next_states = batch

        actual_pred = self._policy_net(torch.cat(previous_states))
        target_pred = self._generate_target_preds(batch, actual_pred, reward)

        loss = self._loss(actual_pred, target_pred)
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

