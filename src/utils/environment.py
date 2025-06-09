# utils/environment.py

# Initialisierung der Umgebung
def initialize_environment(env_mode):
    from envs.grid_environment import GridEnvironment
    from envs.container_environment import ContainerShipEnv

    env = ContainerShipEnv() if env_mode == "container" else GridEnvironment(mode=env_mode)
    grid_size = env.grid_size
    print(f"Umgebung initialisiert: {env_mode}-Modus, Grid-Größe: {grid_size}x{grid_size}")
    return env, grid_size


# Initialisierung der Umgebung für Szenario-Vergleich
def initialize_environment_for_scenario(scenario_config):
    from envs.grid_environment import GridEnvironment
    from envs.container_environment import ContainerShipEnv

    if scenario_config["environment"] == "container":
        env = ContainerShipEnv()
    else:
        env = GridEnvironment(mode=scenario_config["env_mode"])
    return env, env.grid_size