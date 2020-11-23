def _init_summary_writer(self) -> None:
    train_amount = int(
        self._epsilon / (self._epsilon_decay * len(self._memories))
    )
    
    self._summary_writer = SummaryWriter(
        comment=f' - {type(self).__name__} : ' +
                f'alpha={self._alpha}, gamma={self._gamma}, '
                f'train={train_amount}'
    )
    
    self._summary_writer.add_graph(
        self._policy_net, 
        self._current_state[self._memory_index]
    )
    
def _write_train_summary(self) -> None:
    self._summary_writer.add_scalar(
        'Reward', self._rewards_sum[self._memory_index],
        self._episode_cnt[self._memory_index]
    )

    if self._episode_cnt[self._memory_index] % int(
            self._train_amount / 100) == 0:
        for name, param in self._policy_net.named_parameters():
            self._summary_writer.add_histogram(
                name, param, self._episode_cnt[self._memory_index]
            )
            self._summary_writer.add_histogram(
                f'{name}_grad', param.grad, 
                self._episode_cnt[self._memory_index]
            )

