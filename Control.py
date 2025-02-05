import sys
import logging
from connector import RTDEConnect
sys.path.append('..')

class Control():

    def __init__(self, ROBOT_HOST = '169.254.123.187', ROBOT_PORT = 30004, config_filename = 'control_configuration.xml') -> None:
        self.ROBOT_HOST = ROBOT_HOST
        self.ROBOT_PORT = ROBOT_PORT
        self.config_filename = config_filename

        logging.getLogger().setLevel(logging.INFO)


    def setp_to_list(self, output) -> list[float]: 
        setp = [output.input_double_register_0, output.input_double_register_1, output.input_double_register_2,
                output.input_double_register_3, output.input_double_register_4, output.input_double_register_5]
        set_list = [format(elem, '.2f') for elem in setp]

        return [float(x) for x in set_list]
        # Users running 5.11.5 or later can simply return "setp" instead of set_list.
        # return setp

    def sendCoordonates(self, coordonates: list[list[float]]) -> None:
        monitor = RTDEConnect(self.ROBOT_HOST, self.config_filename)
        print("The robot is connected")
        keep_running = True
        index = 0
        ready2Next = False
        
        while keep_running:
            # receive the current state
            state = monitor.receive()

            if state is None:
                break

            print(f"{state.output_int_register_0=}")
            # do something...
            if state.output_int_register_0 != 0 and ready2Next:
                ready2Next = False
                actual = self.setp_to_list(state)
                if actual != [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]:
                    new_coordonates = coordonates[index]
                    index += 1
                else:
                    new_coordonates = coordonates[0]
                monitor.sendall("setp", new_coordonates)
            elif state.output_int_register_0 == 0:
                ready2Next = True

            # kick watchdog
            monitor.send("watchdog", "input_int_register_0", 0)

        monitor.shutdown()
        
        

if __name__ == "__main__":
    con = Control()
    setp1 = [0.250, 0.100, 0.040, 0, 0, 0]
    setp2 = [0.350, 0.100, 0.040, 0, 0, 0]
    setp3 = [0.350, 0, 0.040, 0, 0, 0]
    setp4 = [0.250, 0, 0.040, 0, 0, 0]

    setp = [setp1, setp2, setp3, setp4]
    con.sendCoordonates(setp)