from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingExample:
    character: str
    group: str
    title_class: str
    title_aspect: str

    @property
    def title(self) -> str:
        return f"{self.title_class} of {self.title_aspect}"


TRAINING_EXAMPLES: tuple[TrainingExample, ...] = (
    TrainingExample("John Egbert", "kid", "Heir", "Breath"),
    TrainingExample("Rose Lalonde", "kid", "Seer", "Light"),
    TrainingExample("Dave Strider", "kid", "Knight", "Time"),
    TrainingExample("Jade Harley", "kid", "Witch", "Space"),
    TrainingExample("Jane Crocker", "kid", "Maid", "Life"),
    TrainingExample("Roxy Lalonde", "kid", "Rogue", "Void"),
    TrainingExample("Dirk Strider", "kid", "Prince", "Heart"),
    TrainingExample("Jake English", "kid", "Page", "Hope"),
    TrainingExample("Aradia Megido", "Alternian troll", "Maid", "Time"),
    TrainingExample("Tavros Nitram", "Alternian troll", "Page", "Breath"),
    TrainingExample("Sollux Captor", "Alternian troll", "Mage", "Doom"),
    TrainingExample("Karkat Vantas", "Alternian troll", "Knight", "Blood"),
    TrainingExample("Nepeta Leijon", "Alternian troll", "Rogue", "Heart"),
    TrainingExample("Kanaya Maryam", "Alternian troll", "Sylph", "Space"),
    TrainingExample("Terezi Pyrope", "Alternian troll", "Seer", "Mind"),
    TrainingExample("Vriska Serket", "Alternian troll", "Thief", "Light"),
    TrainingExample("Equius Zahhak", "Alternian troll", "Heir", "Void"),
    TrainingExample("Gamzee Makara", "Alternian troll", "Bard", "Rage"),
    TrainingExample("Eridan Ampora", "Alternian troll", "Prince", "Hope"),
    TrainingExample("Feferi Peixes", "Alternian troll", "Witch", "Life"),
    TrainingExample("Damara Megido", "dancestor", "Witch", "Time"),
    TrainingExample("Rufioh Nitram", "dancestor", "Rogue", "Breath"),
    TrainingExample("Mituna Captor", "dancestor", "Heir", "Doom"),
    TrainingExample("Kankri Vantas", "dancestor", "Seer", "Blood"),
    TrainingExample("Meulin Leijon", "dancestor", "Mage", "Heart"),
    TrainingExample("Porrim Maryam", "dancestor", "Maid", "Space"),
    TrainingExample("Latula Pyrope", "dancestor", "Knight", "Mind"),
    TrainingExample("Aranea Serket", "dancestor", "Sylph", "Light"),
    TrainingExample("Horuss Zahhak", "dancestor", "Page", "Void"),
    TrainingExample("Kurloz Makara", "dancestor", "Prince", "Rage"),
    TrainingExample("Cronus Ampora", "dancestor", "Bard", "Hope"),
    TrainingExample("Meenah Peixes", "dancestor", "Thief", "Life"),
)


def get_training_example(character: str) -> TrainingExample:
    normalized = character.casefold()
    for example in TRAINING_EXAMPLES:
        if example.character.casefold() == normalized:
            return example
    raise ValueError(f"Unknown training character: {character}")
