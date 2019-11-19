from shinra_error import ShinraError


def get_title(html: str) -> str:
    st = html.find("<title>") + len("<title>")
    ed = html.find("</title>")
    if st == -1 or ed == -1:
        raise ShinraError("Title Not found")
    text = html[st:ed]
    ext = text.find(" - Wikipedia Dump")
    if ext != -1:
        return text[:ext]
    else:
        return text
