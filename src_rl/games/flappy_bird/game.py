import pygame


def key_to_action_map(event_type: pygame.event.Event) -> int:
    match event_type:
        case pygame.K_UP | pygame.K_w:
            return 0
        case pygame.K_DOWN | pygame.K_s:
            return 1
        case pygame.K_LEFT | pygame.K_a:
            return 2
        case pygame.K_RIGHT | pygame.K_d:
            return 3
        case _:
            return -1


def map_events(events: list[pygame.event.Event]) -> list[int]:
    actions: list[int] = []
    for event in events:
        if event.type == pygame.KEYDOWN:
            action = key_to_action_map(event.key)
            actions.append(action)
    return actions


def check_if_quit(events: list[pygame.event.Event]) -> bool:
    for event in events:
        if event.type == pygame.QUIT:
            return True
    return False


def gather_events() -> list[pygame.event.Event]:
    events = list(pygame.event.get())
    check_if_quit(events)
    return events


def draw_flappy_bird():
    pass


def run_game() -> None:
    pygame.init()

    running = True
    _ = pygame.display.set_mode((500, 500))

    pygame.display.set_caption("Flappy Bird")
    clock = pygame.time.Clock()

    # state = self.game.processed_state()

    while running:
        events = gather_events()

        actions = map_events(events)
        print(actions)

        if check_if_quit(events):
            break

        pygame.display.flip()
        clock.tick(10)

    pygame.quit()


if __name__ == "__main__":
    run_game()
