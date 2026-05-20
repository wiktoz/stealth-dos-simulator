from dataclasses import dataclass
import json

@dataclass(frozen=True)
class Attacker:
    name: str
    attack_type: str
    intensity: str
    ip: str
    connections: int
    endpoint: str
    start_ts: float
    start_dt: str
    end_ts: float
    end_dt: str
    duration: float


@dataclass
class AttackScenario:
    attackers: list[Attacker]

    @classmethod
    def from_json(cls, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Dict to list[Attacker]
        attackers_list = [Attacker(**item) for item in data]
        return cls(attackers=attackers_list)
