#!/bin/bash
LOCAL_DIR="/mnt/c/Users/roket/Documents/PROJ/ByBit Bot"
REMOTE_HOST="vhizqfra@159.100.6.5"
REMOTE_DIR="/home/vhizqfra/bybitbot"

# Функция для синхронизации файла
sync_file() {
    local_file="$1"
    rel_path="${local_file#$LOCAL_DIR/}"
    remote_file="$REMOTE_DIR/$rel_path"
    
    # Проверяем существование файла на удалённом сервере
    if ssh "$REMOTE_HOST" "[ -f '$remote_file' ]" 2>/dev/null; then
        # Сравниваем размеры файлов
        local_size=$(stat -c%s "$local_file" 2>/dev/null || stat -f%z "$local_file")
        remote_size=$(ssh "$REMOTE_HOST" "stat -c%s '$remote_file' 2>/dev/null || stat -f%z '$remote_file' 2>/dev/null || echo 0")
        
        # Сравниваем даты модификации
        local_mtime=$(stat -c%Y "$local_file" 2>/dev/null || stat -f%m "$local_file")
        remote_mtime=$(ssh "$REMOTE_HOST" "stat -c%Y '$remote_file' 2>/dev/null || stat -f%m '$remote_file' 2>/dev/null || echo 0")
        
        # Если файл изменился, копируем
        if [ "$local_size" -ne "$remote_size" ] || [ "$local_mtime" -gt "$remote_mtime" ]; then
            echo "🔄 Обновляем: $rel_path"
            scp "$local_file" "$REMOTE_HOST:$remote_file"
        else
            echo "✓ Актуально: $rel_path"
        fi
    else
        # Файла нет на удалённом сервере
        echo "➕ Новый файл: $rel_path"
        # Создаём директорию если нужно
        ssh "$REMOTE_HOST" "mkdir -p '$(dirname "$remote_file")'" 2>/dev/null
        scp "$local_file" "$REMOTE_HOST:$remote_file"
    fi
}

echo "🚀 Точечная синхронизация..."

# 1. Синхронизируем файлы в корне проекта (только файлы, не папки)
echo "📁 Корневые файлы проекта:"
find "$LOCAL_DIR" -maxdepth 1 -type f | while read file; do
    sync_file "$file"
done

# 2. Синхронизируем папку scripts (рекурсивно)
if [ -d "$LOCAL_DIR/scripts" ]; then
    echo "📁 Папка scripts:"
    find "$LOCAL_DIR/scripts" -type f | while read file; do
        sync_file "$file"
    done
else
    echo "⚠️ Папка scripts не найдена"
fi

# 3. Синхронизируем папку backups (рекурсивно)
if [ -d "$LOCAL_DIR/backups" ]; then
    echo "📁 Папка backups:"
    find "$LOCAL_DIR/backups" -type f | while read file; do
        sync_file "$file"
    done
else
    echo "⚠️ Папка backups не найдена"
fi

echo "✅ Точечная синхронизация завершена!"