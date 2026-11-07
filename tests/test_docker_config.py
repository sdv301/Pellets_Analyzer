"""Тесты конфигурации Docker и entrypoint.sh.

Проверяет:
1. Unix LF окончания строк (без CRLF \\r\\n) в entrypoint.sh и shell-скриптах.
2. Корректность shebang и команды exec gosu.
3. Наличие автоматической очистки CRLF (sed -i 's/\\r$//') в Dockerfile.
4. Конфигурацию .gitattributes для фиксации LF на Windows.
"""
from pathlib import Path


def test_entrypoint_sh_has_unix_lf_endings():
    entrypoint_path = Path("entrypoint.sh")
    assert entrypoint_path.exists(), "entrypoint.sh не найден!"
    raw = entrypoint_path.read_bytes()
    assert b"\r\n" not in raw, "entrypoint.sh содержит Windows CRLF (\\r\\n), что ломает запуск в Docker!"
    assert raw.startswith(b"#!/bin/bash\n") or raw.startswith(b"#!/bin/sh\n")
    assert b"exec gosu appuser" in raw


def test_dockerfile_normalizes_entrypoint_crlf():
    dockerfile_path = Path("Dockerfile")
    assert dockerfile_path.exists()
    content = dockerfile_path.read_text(encoding="utf-8")
    assert "sed -i 's/\\r$//' /entrypoint.sh" in content or 'sed -i "s/\\r$//" /entrypoint.sh' in content
    assert 'ENTRYPOINT ["/entrypoint.sh"]' in content


def test_gitattributes_enforces_lf_for_shell_scripts():
    gitattributes_path = Path(".gitattributes")
    assert gitattributes_path.exists(), ".gitattributes отсутствует!"
    content = gitattributes_path.read_text(encoding="utf-8")
    assert "*.sh text eol=lf" in content
    assert "Dockerfile text eol=lf" in content
