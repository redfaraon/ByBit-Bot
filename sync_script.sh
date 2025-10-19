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

echo "🚀 Полная синхронизация проекта..."

# Синхронизируем файлы в корневой директории
echo "📁 Корневая директория:"
find "$LOCAL_DIR" -maxdepth 1 -type f | while read file; do
    sync_file "$file"
done

# Синхронизируем все папки (включая scripts и backups)
echo "📁 Все папки:"
find "$LOCAL_DIR" -mindepth 1 -type d | while read dir; do
    folder_name=$(basename "$dir")
    echo "  🔍 Папка: $folder_name"
    find "$dir" -type f | while read file; do
        sync_file "$file"
    done
done

echo "✅ Полная синхронизация завершена!"