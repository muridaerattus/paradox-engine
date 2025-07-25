from fraymotifs.models import Title

def split_titles(title_list: str) -> tuple[list[str], list[str]]:
    """
    Splits a string of titles into lists of individual classes and aspects.
    Handles both single and multiple titles.
    """
    if not title_list:
        return [], []

    titles = [t.strip().lower() for t in title_list.split(',') if t.strip()]
    classes = []
    aspects = []
    valid_classes = ['thief', 'rogue', 'mage', 'seer', 'bard', 'prince', 'witch', 'sylph', 'knight', 'page', 'heir', 'maid']
    valid_aspects = ['doom', 'rage', 'time', 'light', 'void', 'heart', 'mind', 'blood', 'hope', 'space', 'breath', 'life']

    for title in titles:
        parts = [p.strip() for p in title.split(' of ')]
        if len(parts) != 2:
            raise ValueError("Titles must be in the format 'Class of Aspect'.")
        cls, aspect = parts
        if cls not in valid_classes:
            raise ValueError(f"Invalid class: {cls}")
        if aspect not in valid_aspects:
            raise ValueError(f"Invalid aspect: {aspect}")
        classes.append(cls)
        aspects.append(aspect)

    return classes, aspects

def format_titles(titles: list[Title]) -> str:
    """
    Formats a list of Title objects into a string representation.
    """
    if not titles:
        return ""
    return '\n'.join(f"Player {i+1}: {title.title_class} of {title.title_aspect}" for i, title in enumerate(titles))