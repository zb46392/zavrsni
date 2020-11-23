from .subscriber import BaseSubscriber as Subscriber, State


class Publisher:
    def __init__(self) -> None:
        self._subscribers = []

    def attach(self, subscriber: Subscriber) -> None:
        self._subscribers.append(subscriber)

    def detach(self, subscriber: Subscriber) -> None:
        detached = 0

        for i in range(len(self._subscribers)):
            if self._subscribers[i - detached] == subscriber:
                self._subscribers.pop(i - detached)
                detached += 1

    def notify(self, state: State) -> None:
        for subscriber in self._subscribers:
            subscriber.update(state)
