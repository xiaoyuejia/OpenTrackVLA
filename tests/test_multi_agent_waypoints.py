from tools.make_tracking_data import (
    integrate_multi_agent_actions,
    resample_multi_agent_waypoints,
)


def test_nine_actions_produce_origin_plus_nine_waypoints() -> None:
    actions = [[1.0, 0.0, 0.0] for _ in range(9)]
    points = integrate_multi_agent_actions(actions, 0, horizon_steps=9, dt=0.1)
    waypoints, valid = resample_multi_agent_waypoints(points, n_waypoints=10)

    assert len(waypoints) == 10
    assert waypoints[0] == [0.0, 0.0, 0.0]
    assert abs(waypoints[-1][0] - 0.9) < 1e-6
    assert all(valid)
    assert all(waypoints[index][0] < waypoints[index + 1][0] for index in range(9))


def test_partial_horizon_does_not_duplicate_loss_weight() -> None:
    actions = [[1.0, 0.0, 0.0] for _ in range(2)]
    points = integrate_multi_agent_actions(actions, 0, horizon_steps=9, dt=0.1)
    waypoints, valid = resample_multi_agent_waypoints(points, n_waypoints=10)

    assert waypoints[0] == [0.0, 0.0, 0.0]
    assert abs(waypoints[-1][0] - 0.2) < 1e-6
    assert sum(valid) == len(points)
