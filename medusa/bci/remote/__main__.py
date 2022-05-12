
# from medusa.remote.server.configure import *
# configure(sql_host='localhost', sql_user='root', sql_passwd='', db_folder='E://BBDD/medusa-server-databases/')

import sys, os
from medusa.bci.remote.erp_spellers_routes import erp_spellers_routing

from flask import Flask

# Parameters
if len(sys.argv) == 1:
    port = 50000
else:
    port = sys.argv[1]

app = Flask("medusa-server")
app.root_path = os.path.dirname(os.path.abspath(__file__))
app.register_blueprint(erp_spellers_routing)
app.run(host="0.0.0.0", port=port)
