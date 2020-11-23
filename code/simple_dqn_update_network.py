    def _update_policy_net(self, reward: int) -> None:
        self._policy_net.train()

        batch = self._create_batch()
        previous_states = batch[0]
        previous_moves = batch[1]
        previous_possible_actions = batch[2]
        next_states = batch[3]

        actual_pred = self._policy_net(torch.cat(previous_states))
        target_pred = self._generate_target_preds(batch, actual_pred, reward)

        loss = self._loss(actual_pred, target_pred)
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

