import threading, time, socket, struct
import bson
from medusa.bci.erp_storage import *
from medusa.bci.erp_converters import *
from medusa.bci.erp_models import *


def train_model(app_id, info, bin_runs):
    # Unpack the data
    data = list()
    for bin_run in bin_runs:
        # Unpack run
        data.append(RCPRun.unpack_run(bin_run))

    # Response content
    resp_content = dict()

    # Train ERP detection model
    if info["erp_model"] == "None":
        pass
    if info['erp_model'] == 'STDModel':
        erp_model_settings = STDModelSettings()
        erp_model = STDModel()
        erp_model.configure(erp_model_settings)
        erp_model.fit_dataset(data)
        erp_target_codes_per_seq, erp_sel_codes_per_seq, erp_acc_per_seq = erp_model.assess_performance(data)
        # Response content
        resp_content['erp_selected_codes_per_seq'] = erp_sel_codes_per_seq,
        resp_content['erp_accuracy_per_seq'] = erp_acc_per_seq
        resp_content['erp_model'] = pickle.dumps(erp_model)
    else:
        raise Exception('Unknown ERP detection method')

    # Control state detection model
    if info["cs_model"] == "None":
        pass
    if info["cs_model"] == "OSRD":
        cs_model_settings = OSRDModelSettings()
        cs_model = OSRDModel()
        cs_model.configure(cs_model_settings)
        cs_model.fit_dataset(data)
        cs_sel_state_per_seq, cs_acc_per_seq = cs_model.assess_performance(data)
        resp_content["cs_selected_state_per_seq"] = cs_sel_state_per_seq
        resp_content["cs_accuracy_per_seq"] = cs_acc_per_seq
        resp_content['cs_model'] = pickle.dumps(erp_model)
    else:
        raise Exception('Unknown control state detection method')

    return resp_content


class ERPSpellerSession(threading.Thread):
    """
        This class creates a new ERP session for processing trials in real time. It assumes that client is using the
         RCP paradigm, returning the selected row and column along with the probability vector. If an asynchrony model
         is provided, it also returns the control state of the user.
    """
    def __init__(self, unpacker, erp_model, async_model=None, session_timeout=1800, ports=None):
        threading.Thread.__init__(self)
        # Initialize the session socket
        self.session_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Bind socket to port
        if ports is None:
            # Random available port
            self.session_socket.bind(('', 0))
        elif isinstance(ports, int):
            # Bind to port specified in ports
            self.session_socket.bind(('', ports))
        elif isinstance(ports, list):
            # Bind to the first port available from those specified in ports
            counter = 0
            for p in ports:
                try:
                    self.session_socket.bind(('', p))
                    break
                except Exception as e:
                    print(str(e))
                    counter += 1
            if counter > len(ports):
                raise Exception("Cannot find a free port within range")
        else:
            raise Exception("Input ports must be an int or list")
        # Get port
        self.port = self.session_socket.getsockname()[1]
        # Parameters
        self.unpacker = unpacker
        self.erp_model = erp_model
        self.async_model = async_model
        # Timer
        self.session_timeout = session_timeout
        self.last_conn_time = None

    def log_message(self, msg):
        """Log an arbitrary message.

        This is used by all other logging functions.  Override
        it if you have specific logging wishes.

        The first argument, FORMAT, is a format string for the
        message to be logged.  If the format string contains
        any % escapes requiring parameters, they should be
        specified as subsequent arguments (it's just like
        printf!).

        The client ip and current date/time are prefixed to
        every message.

        """

        """Return the current time formatted for logging."""
        monthname = [None,
                     'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        now = time.time()
        year, month, day, hh, mm, ss, x, y, z = time.localtime(now)
        date = "%02d/%3s/%04d %02d:%02d:%02d" % (
            day, monthname[month], year, hh, mm, ss)

        print("\033[92m%s - - [%s] %s\033[0m" % (self.session_socket.getsockname()[0], date, msg))

    def run(self):
        try:
            # Log
            self.log_message("RCP SESSION %s UP" % str(self.port))
            # Session main loop
            request_close = False
            while not request_close:
                self.session_socket.settimeout(self.session_timeout)
                self.session_socket.listen()
                conn, addr = self.session_socket.accept()
                with conn:
                    binary_data = b''
                    while True:
                        part = conn.recv(1024)
                        if not part:
                            if binary_data == b'request_close':
                                request_close = True
                                resp_content = {
                                    "status": 200,
                                    "msg": "Session shutted down"
                                }
                                bin_resp_content = bson.dumps(resp_content)
                                conn.sendall(bin_resp_content)
                                break
                            try:
                                # Unpack the trial
                                fs, n_cha, eeg, times, nseqs, mat_dims, onsets, codes, trials, mat_indices, sequences = self.unpacker.unpack_trial(binary_data)
                                probs = self.erp_model.process_trial(onsets, times, eeg, fs)
                                if self.async_model is not None:
                                    control_state = self.async_model.process_trial(onsets, times, eeg, fs, nseqs)
                                else:
                                    control_state = 1
                                rowcol = rcp_converter(probs, codes, trials, mat_indices, mat_dims)
                                # Trial processed
                                self.log_message("TRIAL PROCESSED (SESSION = %s, LEN = %s B)" % (str(self.port), len(binary_data)))
                                # Headers
                                resp_content = {
                                    "status": 200,
                                    "matrix": rowcol[0],
                                    "sel_row": rowcol[1],
                                    "sel_col": rowcol[2]
                                }
                                bin_resp_content = bson.dumps(resp_content)
                                conn.sendall(bin_resp_content)
                            except Exception as e:
                                resp_content = {
                                    "status": 500,
                                    "msg": str(e)
                                }
                                bin_resp_content = bson.dumps(resp_content)
                                conn.sendall(bin_resp_content)
                            break
                        binary_data += part
            self.log_message("ERP SESSION %s DOWN" % str(self.port))
            self.session_socket.close()
        except socket.timeout:
            self.log_message("ERP SESSION %s TIMEOUT" % str(self.port))
            self.session_socket.close()
