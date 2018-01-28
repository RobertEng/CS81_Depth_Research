def get_version(baseline_data, actions, version):
    assert (len(actions) == 1), "Doesn't make sense to get a certain version across multiple actions"

    # get specific action and version from baseline_data
    action = actions[0]
    if version == 0:
        name = action
    else:
        name = action + " " + str(version)
    action_keys = [k for k in baseline_data.keys() if k[2].split('.')[0] == name]
    action_key = action_keys[0]
    print "Using baseline_data key {}".format(action_key)
    return baseline_data[action_key]