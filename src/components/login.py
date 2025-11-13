# package imports
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Output, Input
from flask_login import current_user, logout_user


login_card = dbc.Card(
    [
        dbc.CardHeader('Login'),
        dbc.CardBody(
            [
                dbc.Input(
                    placeholder='Username',
                    type='text',
                    id='login-username',
                    class_name='mb-2'
                ),
                dbc.Input(
                    placeholder='Password',
                    type='password',
                    id='login-password',
                    class_name='mb-2'
                ),
                dbc.Button(
                    'Login',
                    n_clicks=0,
                    type='submit',
                    id='login-button',
                    class_name='float-end'
                ),
                html.Div(children='', id='login-feedback'),
                html.Br(),
                html.A("Don't have an account? Register here.", href='/register'),
            ]
        )
    ]
)

login_location = dcc.Location(id='url-login')
login_info = html.Div(id='user-status-header')


def logged_in_info(username: str = ''):
    layout = html.Div(
        [
            dbc.Button(
                [html.I(className='fas fa-circle-user fa-xl'), ' ', username],
                id='user-popover',
                outline=True,
                color='light',
                class_name='border-0'
            ),
            dbc.Popover(
                [
                    # dbc.PopoverHeader('Settings'),
                    dbc.PopoverBody(
                        [
                            html.P(username, id='username_display'),
                            dcc.Link(
                                [
                                    html.I(className='fas fa-arrow-right-from-bracket me-1'),
                                    'Logout'
                                ],
                                href='/logout'
                            )
                        ]
                    )
                ],
                target='user-popover',
                trigger='focus',
                placement='bottom'
            )
        ]
    )
    return layout


logged_out_info = dbc.NavItem(
    dbc.NavLink(
        'Login',
        href='/login'
    )
)


@callback(
    Output('user-status-header', 'children'),
    Input('url-login', 'pathname')
)
def update_authentication_status(path):
    from src.extensions import sqlalchemy_db
    
    try:
        logged_in = current_user.is_authenticated
        if path == '/logout' and logged_in:
            logout_user()
            # Clean up the SQLAlchemy session after logout
            sqlalchemy_db.session.remove()
            child = logged_out_info
        elif logged_in:
            child = logged_in_info(current_user.username)
        else:
            child = logged_out_info
        return child
    except Exception as e:
        # Log the error and return logged out info to prevent crash
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error in update_authentication_status: {e}")
        # Clean up the session on error
        try:
            sqlalchemy_db.session.remove()
        except:
            pass
        return logged_out_info
