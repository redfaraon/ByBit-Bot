#!/bin/bash
# ==============================================
# load_env.sh — загрузка и сохранение переменных
# ==============================================

ENV_FILE=".env"
PROFILE_FILE="$HOME/.bashrc"

if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Файл $ENV_FILE не найден"
    exit 1
fi

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "⚠️ Запусти меня так:  source $0"
  exit 0
fi

echo "🔹 Загружаю переменные из $ENV_FILE..."

# Загружаем временно (для текущей сессии)
set -o allexport
source "$ENV_FILE"
set +o allexport

# Проверяем, нужно ли добавить переменные в .bashrc
echo "🔹 Сохраняю переменные в $PROFILE_FILE..."
while IFS='=' read -r key value; do
    # Пропускаем пустые строки и комментарии
    [[ -z "$key" || "$key" =~ ^# ]] && continue

    # Удаляем старую строку, если такая переменная уже есть
    sed -i "/export $key=/d" "$PROFILE_FILE"

    # Добавляем новую строку
    echo "export $key=\"$value\"" >> "$PROFILE_FILE"
done < "$ENV_FILE"

# Применяем изменения
source "$PROFILE_FILE"

echo "✅ Переменные окружения успешно загружены и сохранены!"