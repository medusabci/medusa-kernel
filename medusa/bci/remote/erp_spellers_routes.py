# Built-in packages
import pickle, json

# External packages
from flask import request, Response, Blueprint
import bson

# Internal packages
from medusa.bci.remote import erp_spellers_api

erp_spellers_routing = Blueprint('bci_routing', 'medusa-server', template_folder='templates')


@erp_spellers_routing.route("/erp-speller/<paradigm>/train-model/", methods=['POST'])
def bci_erp_spellers_train_model(paradigm):
    try:
        # Get the info and files attached in the HTTP request
        info = json.loads(request.json)
        files = request_files_to_dict(request)

        # If there is no runs attached, abort
        if "run" not in files:
            raise Exception("No runs attached")
        bin_runs = files['run']

        # If there is no runs attached, abort
        if len(bin_runs) == 0:
            raise Exception("No runs attached")

        # Train models
        resp_content = erp_spellers_api.train_model(paradigm, info, bin_runs)

        # Response request
        bin_resp_content = bson.dumps(resp_content)
        resp = Response(bin_resp_content, status=200, mimetype='application/bson')

    except Exception as e:
        resp_content = {"error": str(e)}
        bin_resp_content = bson.dumps(resp_content)
        resp = Response(bin_resp_content, status=500, mimetype='application/bson')
    return resp


@erp_spellers_routing.route("/erp-spellers/<paradigm>/start-session/", methods=['POST'])
def start_rcp_session(paradigm):
    try:
        # Get the info and files attached in the HTTP request
        info = json.loads(request.json)
        files = request_files_to_dict(request)

        # ERP model is mandatory
        if 'erp_model' in files:
            erp_model = pickle.loads(files['erp_model'])
        else:
            raise Exception("No ERP model attached")

        # Control state model is not mandatory
        if 'cs_model' in files:
            cs_model = pickle.loads(files['erp_model'])
        else:
            cs_model = None

        # Create an ERPSession
        erp_session = erp_spellers_api.ERPSpellerSession(erp_model, cs_model,
                                                         session_timeout=1800,
                                                         ports=list(range(50001, 50007)))
        # Start the session
        erp_session.start()
        # Response content
        resp_content = {"port": erp_session.port}
        bin_resp_content = bson.dumps(resp_content)
        resp = Response(bin_resp_content, status=200, mimetype='application/bson')

    except Exception as e:
        resp_content = {"error": str(e)}
        bin_resp_content = bson.dumps(resp_content)
        resp = Response(bin_resp_content, status=500, mimetype='application/bson')
    return resp
