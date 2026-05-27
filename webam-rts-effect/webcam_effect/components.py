from dataclasses import dataclass

COMPONENT_SEGMENT = "segment"
COMPONENT_CLASSIFY = "classify"
COMPONENT_HAND_TRACK = "hand_track"
COMPONENTS = (COMPONENT_SEGMENT, COMPONENT_CLASSIFY, COMPONENT_HAND_TRACK)


@dataclass(frozen=True)
class ComponentSettings:
    segment: bool = True
    classify: bool = True
    hand_track: bool = True

    def enabled(self, component: str) -> bool:
        return bool(getattr(self, component))


def parse_components(value: str) -> ComponentSettings:
    enabled = {component.strip() for component in value.split(",") if component.strip()}
    unknown = enabled.difference(COMPONENTS)
    if unknown:
        raise ValueError(f"unknown components: {', '.join(sorted(unknown))}")
    return ComponentSettings(
        segment=COMPONENT_SEGMENT in enabled,
        classify=COMPONENT_CLASSIFY in enabled,
        hand_track=COMPONENT_HAND_TRACK in enabled,
    )


def format_components(settings: ComponentSettings) -> str:
    return ",".join(component for component in COMPONENTS if settings.enabled(component))


def select_components_tui(initial: ComponentSettings | None = None) -> ComponentSettings:
    import curses

    settings = initial or ComponentSettings()
    selected = 0

    def run(screen):
        nonlocal selected, settings
        curses.curs_set(0)
        while True:
            screen.erase()
            screen.addstr(0, 0, "Select live components")
            screen.addstr(1, 0, "Space toggle, arrows move, Enter start, q quit")
            for index, component in enumerate(COMPONENTS):
                marker = "[x]" if settings.enabled(component) else "[ ]"
                prefix = ">" if index == selected else " "
                screen.addstr(index + 3, 0, f"{prefix} {marker} {component}")
            screen.refresh()

            key = screen.getch()
            if key in (curses.KEY_UP, ord("k")):
                selected = (selected - 1) % len(COMPONENTS)
            elif key in (curses.KEY_DOWN, ord("j")):
                selected = (selected + 1) % len(COMPONENTS)
            elif key == ord(" "):
                settings = toggle_component(settings, COMPONENTS[selected])
            elif key in (curses.KEY_ENTER, 10, 13):
                return settings
            elif key in (ord("q"), 27):
                raise KeyboardInterrupt

    return curses.wrapper(run)


def toggle_component(settings: ComponentSettings, component: str) -> ComponentSettings:
    values = {name: settings.enabled(name) for name in COMPONENTS}
    values[component] = not values[component]
    return ComponentSettings(**values)
