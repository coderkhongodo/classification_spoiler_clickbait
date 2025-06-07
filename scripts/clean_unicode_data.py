#!/usr/bin/env python3
"""
Script để phát hiện và làm sạch các ký tự Unicode ambiguous trong dữ liệu JSONL
"""

import json
import re
import unicodedata
from pathlib import Path
from collections import defaultdict, Counter

# Danh sách các ký tự Unicode problematic
PROBLEMATIC_CHARS = {
    # Zero-width characters
    '\u200b': '',  # Zero Width Space
    '\u200c': '',  # Zero Width Non-Joiner
    '\u200d': '',  # Zero Width Joiner
    '\u200e': '',  # Left-to-Right Mark
    '\u200f': '',  # Right-to-Left Mark
    '\u2060': '',  # Word Joiner
    '\ufeff': '',  # Byte Order Mark / Zero Width No-Break Space
    
    # Control characters
    '\u0000': '',  # NULL
    '\u0001': '',  # Start of Heading
    '\u0002': '',  # Start of Text
    '\u0003': '',  # End of Text
    '\u0004': '',  # End of Transmission
    '\u0005': '',  # Enquiry
    '\u0006': '',  # Acknowledge
    '\u0007': '',  # Bell
    '\u0008': '',  # Backspace
    '\u000b': '',  # Vertical Tab
    '\u000c': '',  # Form Feed
    '\u000e': '',  # Shift Out
    '\u000f': '',  # Shift In
    '\u0010': '',  # Data Link Escape
    '\u0011': '',  # Device Control One
    '\u0012': '',  # Device Control Two
    '\u0013': '',  # Device Control Three
    '\u0014': '',  # Device Control Four
    '\u0015': '',  # Negative Acknowledge
    '\u0016': '',  # Synchronous Idle
    '\u0017': '',  # End of Transmission Block
    '\u0018': '',  # Cancel
    '\u0019': '',  # End of Medium
    '\u001a': '',  # Substitute
    '\u001b': '',  # Escape
    '\u001c': '',  # File Separator
    '\u001d': '',  # Group Separator
    '\u001e': '',  # Record Separator
    '\u001f': '',  # Unit Separator
    '\u007f': '',  # Delete
    
    # Common homoglyphs (ký tự trông giống nhau)
    '\u2010': '-',  # Hyphen → ASCII Hyphen
    '\u2011': '-',  # Non-breaking Hyphen → ASCII Hyphen
    '\u2012': '-',  # Figure Dash → ASCII Hyphen
    '\u2013': '-',  # En Dash → ASCII Hyphen
    '\u2014': '-',  # Em Dash → ASCII Hyphen
    '\u2015': '-',  # Horizontal Bar → ASCII Hyphen
    '\u2212': '-',  # Minus Sign → ASCII Hyphen
    
    # Quotation marks
    '\u2018': "'",  # Left Single Quotation Mark → ASCII Apostrophe
    '\u2019': "'",  # Right Single Quotation Mark → ASCII Apostrophe
    '\u201a': "'",  # Single Low-9 Quotation Mark → ASCII Apostrophe
    '\u201b': "'",  # Single High-Reversed-9 Quotation Mark → ASCII Apostrophe
    '\u201c': '"',  # Left Double Quotation Mark → ASCII Quote
    '\u201d': '"',  # Right Double Quotation Mark → ASCII Quote
    '\u201e': '"',  # Double Low-9 Quotation Mark → ASCII Quote
    '\u201f': '"',  # Double High-Reversed-9 Quotation Mark → ASCII Quote
    '\u2039': '<',  # Single Left-Pointing Angle Quotation Mark
    '\u203a': '>',  # Single Right-Pointing Angle Quotation Mark
    '\u00ab': '"',  # Left-Pointing Double Angle Quotation Mark
    '\u00bb': '"',  # Right-Pointing Double Angle Quotation Mark
    
    # Spaces
    '\u00a0': ' ',  # Non-Breaking Space → ASCII Space
    '\u2000': ' ',  # En Quad → ASCII Space
    '\u2001': ' ',  # Em Quad → ASCII Space
    '\u2002': ' ',  # En Space → ASCII Space
    '\u2003': ' ',  # Em Space → ASCII Space
    '\u2004': ' ',  # Three-Per-Em Space → ASCII Space
    '\u2005': ' ',  # Four-Per-Em Space → ASCII Space
    '\u2006': ' ',  # Six-Per-Em Space → ASCII Space
    '\u2007': ' ',  # Figure Space → ASCII Space
    '\u2008': ' ',  # Punctuation Space → ASCII Space
    '\u2009': ' ',  # Thin Space → ASCII Space
    '\u200a': ' ',  # Hair Space → ASCII Space
    '\u2028': ' ',  # Line Separator → ASCII Space
    '\u2029': ' ',  # Paragraph Separator → ASCII Space
    '\u202f': ' ',  # Narrow No-Break Space → ASCII Space
    '\u205f': ' ',  # Medium Mathematical Space → ASCII Space
    '\u3000': ' ',  # Ideographic Space → ASCII Space
    
    # Other common problematic characters
    '\u00ad': '',   # Soft Hyphen → Remove
    '\u034f': '',   # Combining Grapheme Joiner → Remove
    '\u061c': '',   # Arabic Letter Mark → Remove
    '\u180e': '',   # Mongolian Vowel Separator → Remove
}

def detect_problematic_chars(text):
    """Phát hiện các ký tự problematic trong text"""
    found_chars = {}
    for char in text:
        if char in PROBLEMATIC_CHARS:
            if char not in found_chars:
                found_chars[char] = 0
            found_chars[char] += 1
    return found_chars

def clean_text(text):
    """Làm sạch text bằng cách thay thế các ký tự problematic"""
    if not isinstance(text, str):
        return text
    
    cleaned = text
    changes_made = {}
    
    # Thay thế các ký tự problematic
    for problematic_char, replacement in PROBLEMATIC_CHARS.items():
        if problematic_char in cleaned:
            count = cleaned.count(problematic_char)
            cleaned = cleaned.replace(problematic_char, replacement)
            changes_made[problematic_char] = count
    
    # Normalize Unicode (NFC normalization)
    cleaned = unicodedata.normalize('NFC', cleaned)
    
    # Remove duplicate spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Strip leading/trailing whitespace
    cleaned = cleaned.strip()
    
    return cleaned, changes_made

def clean_json_object(obj):
    """Làm sạch object JSON recursively"""
    total_changes = defaultdict(int)
    
    if isinstance(obj, str):
        cleaned, changes = clean_text(obj)
        for char, count in changes.items():
            total_changes[char] += count
        return cleaned, total_changes
    
    elif isinstance(obj, list):
        cleaned_list = []
        for item in obj:
            cleaned_item, changes = clean_json_object(item)
            cleaned_list.append(cleaned_item)
            for char, count in changes.items():
                total_changes[char] += count
        return cleaned_list, total_changes
    
    elif isinstance(obj, dict):
        cleaned_dict = {}
        for key, value in obj.items():
            # Clean key
            cleaned_key, key_changes = clean_json_object(key)
            for char, count in key_changes.items():
                total_changes[char] += count
            
            # Clean value
            cleaned_value, value_changes = clean_json_object(value)
            for char, count in value_changes.items():
                total_changes[char] += count
            
            cleaned_dict[cleaned_key] = cleaned_value
        return cleaned_dict, total_changes
    
    else:
        return obj, total_changes

def analyze_file(file_path):
    """Phân tích file để tìm các ký tự problematic"""
    print(f"\n🔍 Phân tích file: {file_path}")
    
    total_chars_found = defaultdict(int)
    total_lines = 0
    problematic_lines = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                total_lines += 1
                try:
                    data = json.loads(line.strip())
                    found_chars = {}
                    
                    # Scan all text fields
                    for key, value in data.items():
                        if isinstance(value, str):
                            chars = detect_problematic_chars(value)
                            for char, count in chars.items():
                                found_chars[char] = found_chars.get(char, 0) + count
                        elif isinstance(value, list):
                            for item in value:
                                if isinstance(item, str):
                                    chars = detect_problematic_chars(item)
                                    for char, count in chars.items():
                                        found_chars[char] = found_chars.get(char, 0) + count
                    
                    if found_chars:
                        problematic_lines += 1
                        for char, count in found_chars.items():
                            total_chars_found[char] += count
                
                except json.JSONDecodeError:
                    print(f"❌ Lỗi JSON ở dòng {line_num}")
    
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")
        return None
    
    print(f"📊 Tổng số dòng: {total_lines}")
    print(f"⚠️  Dòng có vấn đề: {problematic_lines}")
    
    if total_chars_found:
        print("🔍 Ký tự problematic tìm thấy:")
        for char, count in sorted(total_chars_found.items(), key=lambda x: x[1], reverse=True):
            char_name = unicodedata.name(char, f"U+{ord(char):04X}")
            print(f"  '{char}' (U+{ord(char):04X} - {char_name}): {count} lần")
    else:
        print("✅ Không tìm thấy ký tự problematic")
    
    return total_chars_found

def clean_file(input_file, output_file):
    """Làm sạch file và lưu kết quả"""
    print(f"\n🧹 Làm sạch file: {input_file} → {output_file}")
    
    total_changes = defaultdict(int)
    total_lines = 0
    cleaned_lines = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line_num, line in enumerate(f_in, 1):
                total_lines += 1
                try:
                    data = json.loads(line.strip())
                    cleaned_data, changes = clean_json_object(data)
                    
                    if changes:
                        cleaned_lines += 1
                        for char, count in changes.items():
                            total_changes[char] += count
                    
                    # Write cleaned data
                    json_line = json.dumps(cleaned_data, ensure_ascii=False, separators=(',', ':'))
                    f_out.write(json_line + '\n')
                
                except json.JSONDecodeError as e:
                    print(f"❌ Lỗi JSON ở dòng {line_num}: {e}")
                    # Write original line if can't parse
                    f_out.write(line)
    
    except Exception as e:
        print(f"❌ Lỗi xử lý file: {e}")
        return None
    
    print(f"📊 Tổng số dòng xử lý: {total_lines}")
    print(f"🔧 Dòng đã sửa: {cleaned_lines}")
    
    if total_changes:
        print("✨ Thay đổi đã thực hiện:")
        for char, count in sorted(total_changes.items(), key=lambda x: x[1], reverse=True):
            char_name = unicodedata.name(char, f"U+{ord(char):04X}")
            replacement = PROBLEMATIC_CHARS.get(char, '')
            print(f"  '{char}' (U+{ord(char):04X} - {char_name}) → '{replacement}': {count} lần")
    else:
        print("✅ Không có thay đổi nào được thực hiện")
    
    return total_changes

def create_backup(file_path):
    """Tạo backup của file gốc"""
    backup_path = f"{file_path}.backup"
    try:
        import shutil
        shutil.copy2(file_path, backup_path)
        print(f"💾 Đã tạo backup: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"❌ Không thể tạo backup: {e}")
        return None

def main():
    """Hàm chính"""
    print("🔧 CÔNG CỤ LÀM SẠCH KÝ TỰ UNICODE AMBIGUOUS")
    print("=" * 60)
    
    # Files to process
    files_to_process = [
        "data/raw/train.jsonl",
        "data/raw/validation.jsonl"
    ]
    
    # Tạo thư mục backup
    backup_dir = Path("data/backup")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Tạo thư mục cleaned
    cleaned_dir = Path("data/cleaned")
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path in files_to_process:
        if not Path(file_path).exists():
            print(f"⚠️  File không tồn tại: {file_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"📁 Xử lý file: {file_path}")
        print(f"{'='*60}")
        
        # 1. Phân tích file gốc
        problematic_chars = analyze_file(file_path)
        
        if problematic_chars:
            # 2. Tạo backup
            backup_path = f"data/backup/{Path(file_path).name}.backup"
            create_backup(file_path)
            
            # 3. Làm sạch và lưu
            cleaned_path = f"data/cleaned/{Path(file_path).name}"
            changes = clean_file(file_path, cleaned_path)
            
            # 4. Verify cleaned file
            print(f"\n🔍 Kiểm tra file đã làm sạch...")
            verify_chars = analyze_file(cleaned_path)
            
            if not verify_chars:
                print("✅ File đã được làm sạch thành công!")
                
                # Ask user if they want to replace original
                print(f"\n❓ Bạn có muốn thay thế file gốc bằng file đã làm sạch?")
                print(f"   Gốc: {file_path}")
                print(f"   Đã sạch: {cleaned_path}")
                print(f"   Backup: {backup_path}")
                
                choice = input("Nhập 'y' để thay thế, 'n' để giữ nguyên: ").strip().lower()
                if choice == 'y':
                    import shutil
                    shutil.move(cleaned_path, file_path)
                    print(f"✅ Đã thay thế file gốc")
                else:
                    print(f"📁 File đã sạch được lưu tại: {cleaned_path}")
            else:
                print("⚠️  Vẫn còn ký tự problematic sau khi làm sạch!")
        else:
            print("✅ File đã sạch, không cần xử lý")
    
    print(f"\n{'='*60}")
    print("🎉 HOÀN THÀNH LÀM SẠCH DỮ LIỆU!")
    print("📁 Các file backup được lưu trong: data/backup/")
    print("📁 Các file đã sạch được lưu trong: data/cleaned/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 