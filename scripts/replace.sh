#!/bin/bash

# Usage: ./replace_confirm.sh "ancien" "nouveau"

OLD=$1
NEW=$2

# Trouver tous les fichiers texte (on exclut les dossiers cachés comme .git)
files=$(find . -type f -not -path '*/.*')

for file in $files; do
    # Vérifier si le fichier contient le string (évite les binaires)
    if grep -q "$OLD" "$file" 2>/dev/null; then
        echo "--- Fichier: $file ---"
        
        # On crée un fichier temporaire pour le remplacement
        tmp_file=$(mktemp)
        
        # On lit le fichier ligne par ligne
        while IFS= read -r line; do
            if [[ "$line" == *"$OLD"* ]]; then
                # On montre la ligne et on demande confirmation
                echo -e "\033[1;31m- $line\033[0m"
                echo -e "\033[1;32m+ ${line//$OLD/$NEW}\033[0m"
                read -p "Remplacer cette occurrence ? [y/N]: " confirm
                
                if [[ "$confirm" =~ ^[Yy]$ ]]; then
                    echo "${line//$OLD/$NEW}" >> "$tmp_file"
                else
                    echo "$line" >> "$tmp_file"
                fi
            else
                echo "$line" >> "$tmp_file"
            fi
        done < "$file"
        
        mv "$tmp_file" "$file"
    fi
done
