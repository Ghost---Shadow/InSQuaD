from prompt_formatting_strategies.bare import BareStrategy
from prompt_formatting_strategies.q_a_with_new_line import QAWithNewLine


PROMPT_FORMATTING_STRATEGIES_LUT = {
    BareStrategy.NAME: BareStrategy,
    QAWithNewLine.NAME: QAWithNewLine,
}
