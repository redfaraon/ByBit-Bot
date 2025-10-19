#!/bin/bash
LOCAL_DIR="/mnt/c/Users/roket/Documents/PROJ/ByBit Bot"
REMOTE_DIR="/home/vhizqfra/bybitbot"
REMOTE_HOST="vhizqfra@159.100.6.5"

# Создаём временный файл для хранения времени последней синхронизации
LAST_SYNC_FILE="/tmp/last_sync_bybitbot"

# Если файла нет, создаём его с текущим временем
if [ ! -f "$LAST_SYNC_FILE" ]; then
    touch "$LAST_SYNC_FILE"
fi

# Находим файлы, изменённые после последней синхронизации
find "$LOCAL_DIR" -type f -newer "$LAST_SYNC_FILE" | while read file; do
    # Получаем относительный путь
    rel_path="${file#$LOCAL_DIR/}"
    echo "Копируем: $rel_path"
    
    # Создаём директорию на удалённом сервере если нужно
    remote_dir="$REMOTE_DIR/$(dirname "$rel_path")"
    ssh "$REMOTE_HOST" "mkdir -p '$remote_dir'"
    
    # Копируем файл
    scp "$file" "$REMOTE_HOST:$REMOTE_DIR/$rel_path"
done

# Обновляем время последней синхронизации
touch "$LAST_SYNC_FILE"
