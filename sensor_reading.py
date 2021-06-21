import numpy as np


def get_reading(sim, sensor_name):
    id = sim.model.sensor_name2id(sensor_name)
    adr = sim.model.sensor_adr[id]
    dim = sim.model.sensor_dim[id]
    site_id = sim.model.sensor_objid[id]
    site_name = sim.model.site_id2name(site_id)
    or_mat = sim.data.get_site_xmat(site_name)
    sensor_reading = sim.data.sensordata[adr:adr+dim]
    if dim == 3:
        force = np.dot(sensor_reading[0:3], or_mat)
        # x = 2
        # y = 0
        # z = 1
        # force = np.array([force[x], force[y], force[z]])
        sensor_reading = -force.copy()

    return sensor_reading


def sum_of_reading(sim, name_arr):
    reads_sum = 0
    for i in range(len(name_arr)):
        read = get_reading(sim, name_arr[i])
        reads_sum = reads_sum + read

    return reads_sum
