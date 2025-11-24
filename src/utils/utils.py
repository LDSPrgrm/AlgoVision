import json
import re
try:
    from pylatexenc.latexencode import utf8tolatex, UnicodeToLatexEncoder
except:
    print("Warning: Missing pylatexenc, please do pip install pylatexenc")

def _print_response(response_type: str, theorem_name: str, content: str, separator: str = "=" * 50) -> None:
    """Print formatted responses from the video generation process.

    Prints a formatted response with separators and headers for readability.

    Args:
        response_type (str): Type of response (e.g., 'Scene Plan', 'Implementation Plan')
        theorem_name (str): Name of the theorem being processed
        content (str): The content to print
        separator (str, optional): Separator string for visual distinction. Defaults to 50 equals signs.

    Returns:
        None
    """
    print(f"\n{separator}")
    print(f"{response_type} for {theorem_name}:")
    print(f"{separator}\n")
    print(content)
    print(f"\n{separator}")

def _extract_code(response_text: str) -> str:
    """Extract code blocks from a text response.

    Extracts Python code blocks delimited by ```python markers. If no code blocks are found,
    returns the entire response text.

    Args:
        response_text (str): The text response containing code blocks

    Returns:
        str: The extracted code blocks joined by newlines, or the full response if no blocks found
    """
    code = ""
    code_blocks = re.findall(r'```python\n(.*?)\n```', response_text, re.DOTALL)
    if code_blocks:
        code = "\n\n".join(code_blocks)
    elif "```" not in response_text: # if no code block, return the whole response
        code = response_text
    return code 

def extract_json(response: str) -> dict:
    """Extract and parse JSON content from a text response.

    Attempts to parse the response as JSON directly, then tries to extract JSON from code blocks
    if direct parsing fails.

    Args:
        response (str): The text response containing JSON content

    Returns:
        dict: The parsed JSON content as a dictionary, or empty list if parsing fails

    Note:
        Will attempt to parse content between ```json markers first, then between generic ``` markers
    """
    try:
        evaluation_json = json.loads(response)
    except json.JSONDecodeError:
        # If JSON parsing fails, try to extract the content between ```json and ```
        match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
        if not match:
            # If no match for ```json, try to extract content between ``` and ```
            match = re.search(r'```\n(.*?)\n```', response, re.DOTALL)
        
        if match:
            evaluation_content = match.group(1)
            evaluation_json = json.loads(evaluation_content)
        else:
            # return empty list
            evaluation_json = []
            print(f"Warning: Failed to extract valid JSON content from {response}")
    return evaluation_json

def _fix_unicode_to_latex(text: str, parse_unicode: bool = True) -> str:
    """Convert Unicode symbols to LaTeX source code.

    Converts Unicode subscripts and superscripts to LaTeX format, with optional full Unicode parsing.

    Args:
        text (str): The text containing Unicode symbols to convert
        parse_unicode (bool, optional): Whether to perform full Unicode to LaTeX conversion. Defaults to True.

    Returns:
        str: The text with Unicode symbols converted to LaTeX format
    """
    # Map of unicode subscripts to latex format
    subscripts = {
        "₀": "_0", "₁": "_1", "₂": "_2", "₃": "_3", "₄": "_4",
        "₅": "_5", "₆": "_6", "₇": "_7", "₈": "_8", "₉": "_9",
        "₊": "_+", "₋": "_-"
    }
    # Map of unicode superscripts to latex format  
    superscripts = {
        "⁰": "^0", "¹": "^1", "²": "^2", "³": "^3", "⁴": "^4",
        "⁵": "^5", "⁶": "^6", "⁷": "^7", "⁸": "^8", "⁹": "^9",
        "⁺": "^+", "⁻": "^-"
    }

    for unicode_char, latex_format in {**subscripts, **superscripts}.items():
        text = text.replace(unicode_char, latex_format)

    if parse_unicode:
        text = utf8tolatex(text)

    return text

def extract_xml(response: str, tag: str = None) -> str:
    """Extracts XML content from a response.

    This function is designed to robustly extract XML content. It can either pull the
    content from within a specific tag (e.g., <SCENE_OUTLINE>) or from a markdown
    code block (```xml). It includes fallbacks for common LLM response variations.

    Args:
        response (str): The text response potentially containing XML content.
        tag (str, optional): The specific XML tag to extract content from. If provided,
                             this method is prioritized. Defaults to None.

    Returns:
        str: The extracted and cleaned XML content. If no specific content is found,
             it returns the original response.
    """
    # 1. If a specific tag is provided, try to extract its content first.
    if tag:
        match = re.search(f"<{tag}>(.*?)</{tag}>", response, re.DOTALL)
        if match:
            return match.group(1)

    # 2. If no tag is specified or the tag wasn't found, look for a ```xml block.
    match = re.search(r'```xml\n(.*?)\n```', response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 3. Fallback: If no ```xml block, look for a generic ``` block.
    # This is useful if the LLM forgets the 'xml' language identifier.
    match = re.search(r'```\n(.*?)\n```', response, re.DOTALL)
    if match:
        content = match.group(1).strip()
        # Basic check to see if the content looks like XML before returning.
        if content.startswith("<") and content.endswith(">"):
            return content

    # 4. Final Fallback: If no specific content was extracted, return the original response.
    # This handles cases where the LLM returns raw XML without any markdown wrapping.
    return response
