import time

def progress(nsteps, nth_step, time_start, last_message_time, messagePercentDone = [20, 40, 60, 80, 100], active_percent_id = 0, message_period = 30, origin = "solid.a_grid"):
    time_spent = time.time() - time_start
    message_time = last_message_time
    eta = (nsteps / nth_step - 1) * time_spent
    if nth_step / nsteps * 100 > messagePercentDone[active_percent_id]:
        print(origin + ": " + str(messagePercentDone[active_percent_id]) + "% done. ETA: " + str(eta) + " seconds.")
        active_percent_id += 1
        message_time = time.time()

    time_since_last_message = time.time() - last_message_time
    if time_since_last_message > message_period:
        print(origin + ": " + str(nth_step / nsteps * 100) + "% done. ETA: " + str(eta) + " seconds.")
        message_time = time.time()
        
    return active_percent_id, message_time
