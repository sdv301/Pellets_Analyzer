import pytest
import sqlite3
from unittest.mock import patch, MagicMock

from app.auth.auth import (
    hash_password, check_password, create_user, get_user_by_email,
    get_user_by_id, authenticate_user, update_user_profile,
    request_password_reset, reset_password, log_activity,
    get_activity_logs, change_password, init_auth_tables, get_user_stats
)


@pytest.fixture
def auth_db(test_db_path):
    init_auth_tables(test_db_path)
    return test_db_path


class TestPasswordHashing:
    def test_hash_password_returns_string(self):
        hashed = hash_password("secret123")
        assert isinstance(hashed, str)
        assert len(hashed) > 20

    def test_check_password_correct(self):
        hashed = hash_password("secret123")
        assert check_password("secret123", hashed) is True

    def test_check_password_incorrect(self):
        hashed = hash_password("secret123")
        assert check_password("wrong123", hashed) is False


class TestUserCRUD:
    def test_create_user(self, auth_db):
        result = create_user(
            auth_db, "john_doe", "john@example.com", "pass123", "John Doe", "user"
        )
        assert result["success"] is True
        assert "user_id" in result
        assert "verification_token" in result

    def test_create_user_duplicate_email(self, auth_db):
        create_user(auth_db, "john1", "john@example.com", "pass", "John", "user")
        result = create_user(auth_db, "john2", "john@example.com", "pass", "John", "user")
        assert result["success"] is False
        assert "email" in result["error"].lower()

    def test_get_user_by_id(self, auth_db):
        create_res = create_user(
            auth_db, "jane_doe", "jane@example.com", "pass123", "Jane Doe", "user"
        )
        user = get_user_by_id(auth_db, create_res["user_id"])
        assert user is not None
        assert user["username"] == "jane_doe"
        assert user["email"] == "jane@example.com"
        assert user["role_name"] == "user"

    def test_get_user_by_email(self, auth_db):
        create_user(auth_db, "bob", "bob@example.com", "pass", "Bob", "user")
        user = get_user_by_email(auth_db, "bob@example.com")
        assert user is not None
        assert user["username"] == "bob"

    def test_update_user_profile(self, auth_db):
        create_res = create_user(
            auth_db, "alice", "alice@example.com", "pass", "Alice", "user"
        )
        uid = create_res["user_id"]
        
        result = update_user_profile(
            auth_db, uid, full_name="Alice Smith", company="Acme Corp"
        )
        assert result["success"] is True

        user = get_user_by_id(auth_db, uid)
        assert user["full_name"] == "Alice Smith"
        assert user["company"] == "Acme Corp"


class TestAuthentication:
    def test_authenticate_success(self, auth_db):
        create_user(auth_db, "testuser", "test@test.com", "mypass", "Test", "user")
        
        # Verify the user manually for testing
        conn = sqlite3.connect(auth_db)
        conn.execute("UPDATE users SET is_verified = 1 WHERE email = 'test@test.com'")
        conn.commit()
        conn.close()

        result = authenticate_user(auth_db, "test@test.com", "mypass")
        assert result["success"] is True
        assert result["user"]["username"] == "testuser"

    def test_authenticate_wrong_password(self, auth_db):
        create_user(auth_db, "user2", "user2@test.com", "mypass", "User2", "user")
        
        conn = sqlite3.connect(auth_db)
        conn.execute("UPDATE users SET is_verified = 1 WHERE email = 'user2@test.com'")
        conn.commit()
        conn.close()

        result = authenticate_user(auth_db, "user2@test.com", "wrongpass")
        assert result["success"] is False
        assert result.get("error")

    def test_authenticate_not_verified(self, auth_db):
        create_user(auth_db, "user3", "user3@test.com", "mypass", "User3", "user")
        
        result = authenticate_user(auth_db, "user3@test.com", "mypass")
        assert result["success"] is False
        assert result.get("error")


class TestPasswordManagement:
    def test_change_password(self, auth_db):
        create_res = create_user(auth_db, "user4", "user4@test.com", "oldpass", "User4", "user")
        uid = create_res["user_id"]
        
        result = change_password(auth_db, uid, "oldpass", "newpass")
        assert result["success"] is True

        conn = sqlite3.connect(auth_db)
        conn.execute("UPDATE users SET is_verified = 1 WHERE id = ?", (uid,))
        conn.commit()
        conn.close()

        auth_res = authenticate_user(auth_db, "user4@test.com", "newpass")
        assert auth_res["success"] is True

    def test_request_password_reset(self, auth_db):
        create_user(auth_db, "user5", "user5@test.com", "pass", "User5", "user")
        
        result = request_password_reset(auth_db, "user5@test.com")
        assert result["success"] is True
        assert "reset_token" in result

    def test_reset_password(self, auth_db):
        create_user(auth_db, "user6", "user6@test.com", "pass", "User6", "user")
        req_res = request_password_reset(auth_db, "user6@test.com")
        token = req_res["reset_token"]
        
        reset_res = reset_password(auth_db, token, "newpass123")
        assert reset_res["success"] is True

        conn = sqlite3.connect(auth_db)
        conn.execute("UPDATE users SET is_verified = 1 WHERE email = 'user6@test.com'")
        conn.commit()
        conn.close()

        auth_res = authenticate_user(auth_db, "user6@test.com", "newpass123")
        assert auth_res["success"] is True


class TestActivityLogging:
    @patch('app.auth.auth.get_real_ip')
    def test_log_activity(self, mock_ip, auth_db):
        mock_ip.return_value = "192.168.1.1"
        create_res = create_user(auth_db, "logger", "logger@test.com", "pass", "Logger", "user")
        uid = create_res["user_id"]
        
        log_activity(auth_db, uid, "login", "User logged in")
        
        logs = get_activity_logs(auth_db, uid, limit=10)
        assert len(logs) > 0
        assert any(log["action"] == "login" for log in logs)

    def test_get_user_stats(self, auth_db):
        create_user(auth_db, "user7", "user7@test.com", "pass", "User", "user")
        create_user(auth_db, "admin1", "admin1@test.com", "pass", "Admin", "admin")
        
        stats = get_user_stats(auth_db)
        assert stats["total_users"] >= 2
        assert stats["active_users"] >= 2
