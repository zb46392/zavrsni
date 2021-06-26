from torch.utils.tensorboard import SummaryWriter

def _init_summary_writer(self) -> None:
    dummy_input = self._generate_dummy_input()
    log_dir_path = self._generate_log_dir_path()

    self._summary_writer = SummaryWriter(log_dir=str(log_dir_path))
    self._summary_writer.add_graph(self._model, dummy_input)

def monitor(self) -> None:
    progress = self._progress
    self._summary_writer.add_scalar('Progress', progress, self._step)

    for name, param in self._model.named_parameters():
        self._summary_writer.add_histogram(name, param, self._step)
        self._summary_writer.add_histogram(f'{name}_grad', param.grad, self._step)

    self._step += 1

def close(self) -> None:
    self._summary_writer.close()

