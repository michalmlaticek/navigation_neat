# if self.simulation.simulation_conf.animate:
        #     fig = plt.Figure()
        #     im = plt.imshow(viz.get_image(simulation.robot_bodies, simulation.sensor_lines, simulation_conf.map))
        #     simulation.simulate(simulation_conf.step_count, step_callback=save_and_draw,
        #                         callback_args=(gif_frames, simulation, im))
        # else:
        #     simulation.simulate(simulation_conf.step_count, step_callback=save,
        #                         callback_args=(gif_frames, simulation))


def save(frames, sim):
    frame = viz.get_image(sim.robot_bodies, sim.sensor_lines, sim.map)
    frames.append(viz.to_zxy(frame))
    return frame


def save_and_draw(frames, sim, img):
    frame = save(frames, sim)
    img.set_data(frame)
    plt.draw()
    plt.pause(0.0001)