#!/usr/bin/env bash
# Сдвинуть пин субмодуля lab/ на тег или коммит.
# Зелёные тесты, сравнение со снимком и пересборка артефакта — затем git add.
# Иначе субмодуль возвращается на прежний коммит. Коммит продукта делает владелец.
#
#   bash scripts/update_lab.sh lab-v0.1.0
#   bash scripts/update_lab.sh 959e545d1649c1acfa4ef108a64a45d843ca20f4
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: scripts/update_lab.sh <tag-or-commit>" >&2
  exit 2
fi
REF="$1"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

if [[ ! -e lab/.git ]]; then
  git submodule update --init lab
fi

OLD="$(git rev-parse HEAD:lab 2>/dev/null || git -C lab rev-parse HEAD)"
echo "текущий пин: $OLD"
echo "запрошено: $REF"

cleanup_failed() {
  echo "обновление не принято, возвращаю lab на $OLD" >&2
  git -C lab checkout --detach "$OLD" || git -C lab checkout "$OLD"
}

git -C lab fetch --tags origin
git -C lab checkout --detach "$REF"
NEW="$(git -C lab rev-parse HEAD)"
echo "lab HEAD: $NEW"

export PELLET_LAB_PATH="$ROOT/lab"
export PYTHONPATH="$ROOT/lab${PYTHONPATH:+:$PYTHONPATH}"
export PELLET_MODEL_DIR="$ROOT/models/v2"
export PELLET_LAB_COMMIT="$NEW"

set +e
python -m pytest "$ROOT/lab/tests/test_public_api.py" -q --tb=short
lab_status=$?
python -m pytest \
  "$ROOT/tests/test_v2_parser.py" \
  "$ROOT/tests/test_v2_contract.py" \
  "$ROOT/tests/test_v2_switch.py" \
  "$ROOT/tests/test_v2_auth.py" \
  "$ROOT/tests/test_v2_evaluation.py" \
  "$ROOT/tests/test_v2_submodule.py" \
  -q --tb=short
prod_status=$?
set -e

if [[ $lab_status -ne 0 || $prod_status -ne 0 ]]; then
  cleanup_failed
  exit 1
fi

if ! python "$ROOT/scripts/build_v2_artifact.py" --out "$PELLET_MODEL_DIR"; then
  cleanup_failed
  exit 1
fi

if ! python "$ROOT/scripts/compare_v2_baseline.py"; then
  echo "Если смена прогноза осознанная, обновите tests/fixtures/v2_baseline.json отдельным коммитом и повторите команду." >&2
  cleanup_failed
  exit 1
fi

printf '%s\n' "$NEW" > "$ROOT/app/services/ml_v2/PIN"
git add lab app/services/ml_v2/PIN
echo "Пин подготовлен: $OLD -> $NEW"
echo "В индекс добавлены gitlink lab и app/services/ml_v2/PIN. Коммит и PR — вручную."
echo "Откат до коммита: git checkout $OLD -- lab && git checkout HEAD -- app/services/ml_v2/PIN"
echo "Либо откатить весь bump-коммит после того, как он создан: git revert <sha>"
