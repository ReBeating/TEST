import json
import subprocess
import os
import tempfile
from typing import Dict, Any, Optional
import tree_sitter_cpp
import tree_sitter_c
from tree_sitter import Language, Parser
import shutil

class CtagsParser:
    """基于 Universal Ctags 的代码解析器"""
    
    @staticmethod
    def _run_ctags(target_path: str) -> str:
        """运行 ctags 命令并返回 JSON 输出"""
        # --fields=+ne: n=line number, e=end line number
        # --c-kinds=fsdue: f=function, s=struct, d=macro, u=union, e=enum
        cmd = [
            "ctags", "--fields=+ne", "--output-format=json", "--c-kinds=fsdue",
            "-o", "-", target_path
        ]
        try:
            # 增加 errors='ignore' 防止解码非 utf-8 文件报错
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8', errors='replace')
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"[Ctags] Error processing {target_path}: {e}")
            return ""
        except FileNotFoundError:
            print("[Ctags] Error: 'ctags' command not found. Please install universal-ctags.")
            return ""

    @staticmethod
    def parse_file(file_path: str) -> Dict[str, Dict[str, Any]]:
        """
        解析指定文件
        Return: {'functions': {...}, 'structs': {...}, 'macros': {...}}
        """
        if not os.path.exists(file_path):
            return {'functions': {}, 'structs': {}, 'macros': {}}

        # 读取源码用于切片
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                code = f.read()
        except Exception as e:
            print(f"[Ctags] Read file error: {e}")
            return {'functions': {}, 'structs': {}, 'macros': {}}

        ctags_output = CtagsParser._run_ctags(file_path)
        return CtagsParser._process_tags(ctags_output, code)

    @staticmethod
    def parse_code(code: str, suffix='.c') -> Dict[str, Dict[str, Any]]:
        """解析内存中的代码字符串"""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=suffix, delete=True) as tmp_file:
            tmp_file.write(code)
            tmp_file.flush()
            ctags_output = CtagsParser._run_ctags(tmp_file.name)
            return CtagsParser._process_tags(ctags_output, code)

    @staticmethod
    def _process_tags(ctags_output: str, code: str) -> Dict[str, Dict[str, Any]]:
        """内部逻辑：处理 ctags 的 JSON 输出并切片"""
        lines = code.replace('\x0c', ' ').splitlines() 
        parse_result = {'functions': {}, 'structs': {}, 'macros': {}}
        
        for line in ctags_output.splitlines():
            try:
                tag = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            kind = tag.get('kind', '')
            name = tag.get("name", "")
            start = tag.get("line", 0)
            end = tag.get("end", start) # 如果没有 end，默认单行
            
            # ctags 是 1-based，python list 是 0-based
            # 你的逻辑：如果上一行也是名字开头（可能这行是返回类型？），前移
            # 这里保留你的 heuristic，但加个边界检查
            idx_start = max(0, start - 1)
            
            # 尝试向上修正 (处理 return type 换行的情况)
            # 例如: 
            # void
            # my_func() { ... }
            if kind == 'function' and idx_start > 0:
                prev_line = lines[idx_start - 1].strip()
                # 简单的启发式：如果上一行不是以分号或右大括号结尾，可能属于函数头的一部分
                if prev_line and not prev_line.endswith((';', '}')) and not prev_line.startswith(('#', '//')):
                     idx_start -= 1

            idx_end = end # slice 是左闭右开，所以 end 不需要 -1
            
            tag_code = "\n".join(lines[idx_start:idx_end])
            
            item = {
                'code': tag_code, 
                'start_line': idx_start + 1, # 转回 1-based 用于显示
                'end_line': idx_end
            }

            if kind == 'function':
                parse_result['functions'][name] = item
            elif kind in ('struct', 'union', 'enum'):
                parse_result['structs'][name] = item
            elif kind == 'macro':
                parse_result['macros'][name] = item
                
        return parse_result

def remove_comments(code: str, lang: str = "cpp") -> str:
    """
    直接去除代码中的注释 (支持 C/C++)。
    自动处理 tree-sitter 新旧版本 API 差异。
    """
    # 1. 初始化语言和解析器
    if lang == "c":
        language = Language(tree_sitter_c.language())
    elif lang == "cpp":
        language = Language(tree_sitter_cpp.language())
    else:
        raise ValueError(f"Unsupported language: {lang}")

    parser = Parser(language)
    
    # 2. 解析代码
    # tree-sitter 处理字节偏移，所以先 encode
    code_bytes = code.encode('utf-8', errors='replace')
    tree = parser.parse(code_bytes)
    
    # 3. 查找所有注释节点
    # 使用 Query API 查找，比递归遍历快且安全
    query = language.query("(comment) @comment")
    captures = query.captures(tree.root_node)

    # 4. 获取注释的字节范围 (start_byte, end_byte)
    # 【关键】兼容 tree-sitter 新版(dict) 和 旧版(list) 返回格式
    ranges = []
    if isinstance(captures, dict):
        # 新版: {'comment': [Node, Node, ...]}
        for nodes in captures.values():
            for node in nodes:
                ranges.append((node.start_byte, node.end_byte))
    else:
        # 旧版: [(Node, 'comment'), ...]
        for node, _ in captures:
            ranges.append((node.start_byte, node.end_byte))
    
    # 如果没有注释，直接返回原代码
    if not ranges:
        return code

    # 5. 执行删除
    # 按 start_byte 倒序排列，确保前面的删除不会影响后面的偏移量
    ranges.sort(key=lambda x: x[0], reverse=True)
    
    for start, end in ranges:
        # 将注释部分替换为空字节
        code_bytes = code_bytes[:start] + code_bytes[end:]
        
    return code_bytes.decode('utf-8', errors='replace')


def indent_code(code: str) -> str:
    # 1. 检查 indent 工具是否存在
    if not shutil.which("indent"):
        return code

    # 参数列表（建议根据需求调整，这里保留了你原本的参数）
    indent_args = '-nbad -bap -nbc -bbo -hnl -br -brs -c33 -cd33 -ncdb -ce -ci4 -cli0 -d0 -di1 -nfc1 -i8 -ip0 -l9999 -lp -npcs -nprs -npsl -sai -saf -saw -ncs -nsc -sob -nfca -cp33 -ss -ts8 -il1'
    
    # 2. 使用 split() 而不是 split(' ') 以避免多余空格导致的问题
    cmd = ["indent"] + indent_args.split() + ["-"]

    try:
        result = subprocess.run(
            cmd,
            input=code.encode("utf-8", errors='replace'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # 设置超时防止死锁（可选）
            timeout=10 
        )

        # 3. 关键：检查返回码
        if result.returncode != 0:
            error_msg = result.stderr.decode("utf-8", errors='replace').strip()
            # 格式化失败时，务必返回原始内容，防止代码丢失
            return code

        # 4. 解码输出
        return result.stdout.decode("utf-8", errors='replace')

    except subprocess.TimeoutExpired:
        return code
    except Exception as e:
        return code