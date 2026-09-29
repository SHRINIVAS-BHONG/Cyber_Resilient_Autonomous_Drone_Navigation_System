"""
Attack Simulator Package.
"""

from attack_simulator.attack_engine import AttackSimulationEngine

try:
    from attack_simulator.attack_injector_node import AttackInjectorNode
    __all__ = ["AttackSimulationEngine", "AttackInjectorNode"]
except ImportError:
    __all__ = ["AttackSimulationEngine"]
