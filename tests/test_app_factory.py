import pytest
from app import create_app
import tempfile
import os


@pytest.fixture
def test_app():
    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    
    app = create_app({
        'TESTING': True,
        'SECRET_KEY': 'test-key-123',
        'DATABASE_PATH': db_path
    })
    
    yield app
    
    os.close(db_fd)
    os.unlink(db_path)


class TestCreateApp:
    def test_create_app_config(self, test_app):
        assert test_app.config['TESTING'] is True
        assert test_app.config['SECRET_KEY'] == 'test-key-123'
        assert test_app.config['SESSION_TYPE'] == 'filesystem'
        assert 'UPLOAD_FOLDER' in test_app.config

    def test_blueprints_registered(self, test_app):
        blueprints = test_app.blueprints.keys()
        assert 'main' in blueprints
        assert 'auth' in blueprints
        assert 'compare' in blueprints
        assert 'economics' in blueprints
        assert 'graphs' in blueprints
        assert 'ml' in blueprints
        assert 'admin' in blueprints

    def test_extensions_initialized(self, test_app):
        assert 'AI_ML_ANALYZER' in test_app.config
        assert test_app.config['AI_ML_ANALYZER'] is not None


class TestErrorHandlers:
    def test_404_handler(self, test_app):
        client = test_app.test_client()
        response = client.get('/non_existent_route_12345')
        assert response.status_code == 404
        assert b"404" in response.data

    def test_403_handler(self, test_app):
        # Admin dashboard should return 403 or redirect for anonymous users
        client = test_app.test_client()
        response = client.get('/admin/dashboard')
        assert response.status_code in [302, 401, 403]


class TestContextProcessor:
    def test_inject_user_anonymous(self, test_app):
        with test_app.test_request_context('/'):
            # Manually invoke the context processor
            context = test_app.jinja_env.globals.copy()
            for cp in test_app.template_context_processors[None]:
                context.update(cp())
            
            assert 'current_user' in context
            assert context['current_user']['is_authenticated'] is False

    def test_inject_user_authenticated(self, test_app):
        client = test_app.test_client()
        with client.session_transaction() as sess:
            sess['user_id'] = 1
            sess['username'] = 'tester'
            sess['email'] = 'test@example.com'
            sess['role_name'] = 'admin'
            sess['full_name'] = 'Tester'
            
        with test_app.test_request_context('/'):
            # We need to simulate the request context with session data
            from flask import session
            session['user_id'] = 1
            session['username'] = 'tester'
            session['email'] = 'test@example.com'
            session['role_name'] = 'admin'
            session['full_name'] = 'Tester'
            
            context = test_app.jinja_env.globals.copy()
            for cp in test_app.template_context_processors[None]:
                context.update(cp())
            
            assert 'current_user' in context
            assert context['current_user']['is_authenticated'] is True
            assert context['current_user']['username'] == 'tester'
            assert context['current_user']['role_name'] == 'admin'
