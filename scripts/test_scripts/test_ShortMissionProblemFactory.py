from ocean_navigation_simulator.problem_factories.ShortMissionProblemFactory import ShortMissionProblemFactory


for i in range(1):
    problem_factory = ShortMissionProblemFactory(
        scenario_name='gulf_of_mexico_HYCOM_hindcast',
        seed=i,
        verbose=1,
    )
    problems = problem_factory.generate_batch(4)
    problem_factory.plot_batch(4, filename='/seaweed-storage/tmp/animation.gif')
    problem_factory.hindcast_planner.save_plan('/seaweed-storage/tmp/planner/')

