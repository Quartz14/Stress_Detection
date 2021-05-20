from flask import Flask

def init_app():
    app = Flask(__name__)#, instance_relative_config=False)
    #app.config.from_object('config.Config')

    with app.app_context():
        from . import routes
        #from .plotlydash.dashboard import init_dashboard
        from .plotlydash.dashboard2 import init_dashboard
        from .plotlydash.dashmodel import init_dashmodel
        from .plotlydash.dashlivegraphs import init_dashlive
        from .plotlydash.dashprocess import init_dashprocess




        app = init_dashboard(app)
        app = init_dashmodel(app)
        app = init_dashprocess(app)
        app = init_dashlive(app)
        return app
