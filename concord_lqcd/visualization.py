"""Physicist-facing HTML and LaTeX views of a ConCorD job.

This presentation layer reads a validated job and symbolic operator factory,
plus an optional compiled plan and versioned run record. It never compiles a
plan, loads lattice tensors, or runs contractions. The generated HTML is
self-contained and uses browser MathML while retaining the exact LaTeX source
beneath each expression.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import html
import json
import os
from pathlib import Path

from concord_lqcd.core.color import ColorDelta, ColorEpsilon
from concord_lqcd.core.correlator import CorrMatrix, Correlator
from concord_lqcd.core.expr import Expr, Product
from concord_lqcd.core.fields import QuarkField
from concord_lqcd.core.gamma import GammaMatrix
from concord_lqcd.core.gauge import GaugeLink
from concord_lqcd.core.indices import COLOR, POSITION, SPIN, Index
from concord_lqcd.core.operator import Operator
from concord_lqcd.core.operator_expression import OperatorCombination
from concord_lqcd.core.operations import (
    Apply,
    CovariantDerivative,
    ParallelTransport,
    Projection,
    Shift,
)
from concord_lqcd.core.propagator import Propagator
from concord_lqcd.job.loader import OperatorCollection
from concord_lqcd.job.plan import ConcordPlan
from concord_lqcd.job.run_record import JobRunRecord
from concord_lqcd.job.schema import JobSpec


class PhysicsVisualizationError(ValueError):
    """Raised when a physics visualization cannot be represented safely."""


@dataclass(frozen=True)
class Formula:
    """One mathematical expression in LaTeX and browser-native MathML."""

    latex: str
    mathml: str


_SPIN_SYMBOLS = (
    (r"\alpha", "α"),
    (r"\beta", "β"),
    (r"\gamma", "γ"),
    (r"\delta", "δ"),
    (r"\rho", "ρ"),
    (r"\sigma", "σ"),
    (r"\lambda", "λ"),
    (r"\kappa", "κ"),
    (r"\mu", "μ"),
    (r"\nu", "ν"),
)
_COLOR_SYMBOLS = tuple((letter, letter) for letter in "abcdefgh")
_POSITION_SYMBOLS = tuple(
    (rf"\mathbf{{{letter}}}", letter) for letter in ("x", "y", "z", "w")
)


def _escape_latex_text(value: str) -> str:
    replacements = {
        "\\": r"\backslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\sim{}",
        "^": r"\hat{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _roman(value: str) -> Formula:
    escaped = _escape_latex_text(value)
    return Formula(
        latex=rf"\mathrm{{{escaped}}}",
        mathml=f'<mi mathvariant="normal">{html.escape(value)}</mi>',
    )


def _row(*parts: Formula, latex_separator: str = "") -> Formula:
    return Formula(
        latex=latex_separator.join(part.latex for part in parts),
        mathml="<mrow>" + "".join(part.mathml for part in parts) + "</mrow>",
    )


def _operator_token(value: str) -> Formula:
    return Formula(
        latex=value,
        mathml=f"<mo>{html.escape(value)}</mo>",
    )


def _parenthesized(value: Formula) -> Formula:
    return Formula(
        latex=rf"\left({value.latex}\right)",
        mathml=(
            '<mrow><mo stretchy="true">(</mo>'
            f"{value.mathml}"
            '<mo stretchy="true">)</mo></mrow>'
        ),
    )


def _subscript(base: Formula, subscript: Formula) -> Formula:
    return Formula(
        latex=rf"{base.latex}_{{{subscript.latex}}}",
        mathml=f"<msub>{base.mathml}{subscript.mathml}</msub>",
    )


def _superscript(base: Formula, superscript: Formula) -> Formula:
    return Formula(
        latex=rf"{base.latex}^{{{superscript.latex}}}",
        mathml=f"<msup>{base.mathml}{superscript.mathml}</msup>",
    )


def _subsup(base: Formula, subscript: Formula, superscript: Formula) -> Formula:
    return Formula(
        latex=rf"{base.latex}_{{{subscript.latex}}}^{{{superscript.latex}}}",
        mathml=(
            f"<msubsup>{base.mathml}{subscript.mathml}"
            f"{superscript.mathml}</msubsup>"
        ),
    )


class _FormulaContext:
    def __init__(self) -> None:
        self._symbols: dict[Index, Formula] = {}
        self._counts = {SPIN: 0, COLOR: 0, POSITION: 0}

    def index(self, index: Index) -> Formula:
        existing = self._symbols.get(index)
        if existing is not None:
            return existing

        count = self._counts[index.space]
        self._counts[index.space] = count + 1
        if index.space == SPIN:
            symbol = self._indexed_symbol(_SPIN_SYMBOLS, count, "s")
        elif index.space == COLOR:
            symbol = self._indexed_symbol(_COLOR_SYMBOLS, count, "c")
        elif index.space == POSITION:
            symbol = self._indexed_symbol(_POSITION_SYMBOLS, count, "x", bold=True)
        else:  # pragma: no cover - Index validates its space
            raise PhysicsVisualizationError(f"unsupported index space {index.space!r}")
        self._symbols[index] = symbol
        return symbol

    def positions(self, expression: Expr) -> tuple[tuple[Index, Formula], ...]:
        unique: list[Index] = []
        for index in expression.indices:
            if index.space == POSITION and index not in unique:
                unique.append(index)
        return tuple((index, self.index(index)) for index in unique)

    @staticmethod
    def _indexed_symbol(
        symbols: Sequence[tuple[str, str]],
        count: int,
        fallback: str,
        *,
        bold: bool = False,
    ) -> Formula:
        if count < len(symbols):
            latex, text = symbols[count]
            variant = ' mathvariant="bold"' if bold else ""
            return Formula(latex=latex, mathml=f"<mi{variant}>{text}</mi>")
        ordinal = count + 1
        base_latex = rf"\mathbf{{{fallback}}}" if bold else fallback
        base_mathml = (
            f'<mi mathvariant="bold">{fallback}</mi>'
            if bold
            else f"<mi>{fallback}</mi>"
        )
        return Formula(
            latex=rf"{base_latex}_{{{ordinal}}}",
            mathml=f"<msub>{base_mathml}<mn>{ordinal}</mn></msub>",
        )


def _gamma_symbol(kind: str) -> Formula:
    daggered = kind.endswith("_dag")
    base_kind = kind.removesuffix("_dag") if daggered else kind
    gamma_map = {
        "g5": 5,
        "gamma5": 5,
        "P": 5,
        "g1": 1,
        "gammaX": 1,
        "V1": 1,
        "g2": 2,
        "gammaY": 2,
        "V2": 2,
        "g3": 3,
        "gammaZ": 3,
        "V3": 3,
        "g4": 4,
        "gammaT": 4,
        "V4": 4,
    }
    if base_kind in gamma_map:
        number = gamma_map[base_kind]
        symbol = Formula(
            latex=rf"\gamma_{{{number}}}",
            mathml=f"<msub><mi>γ</mi><mn>{number}</mn></msub>",
        )
    elif base_kind in {"Identity", "gamma_id", "S"}:
        symbol = Formula(
            latex=r"\mathbf{1}",
            mathml='<mn mathvariant="bold">1</mn>',
        )
    elif base_kind == "C":
        symbol = Formula(latex="C", mathml="<mi>C</mi>")
    elif base_kind.startswith("Cg") and base_kind[2:].isdigit():
        number = int(base_kind[2:])
        symbol = _row(
            Formula(latex="C", mathml="<mi>C</mi>"),
            Formula(
                latex=rf"\gamma_{{{number}}}",
                mathml=f"<msub><mi>γ</mi><mn>{number}</mn></msub>",
            ),
            latex_separator=r"\,",
        )
    elif base_kind in {"Cg5", "Cg5Upper"}:
        symbol = _row(
            Formula(latex="C", mathml="<mi>C</mi>"),
            Formula(
                latex=r"\gamma_{5}",
                mathml="<msub><mi>γ</mi><mn>5</mn></msub>",
            ),
            latex_separator=r"\,",
        )
        if base_kind.endswith("Upper"):
            symbol = _superscript(symbol, _roman("upper"))
    elif base_kind in {"Pplus", "Pminus", "PplusUpper"}:
        sign = "+" if "plus" in base_kind else "−"
        latex_sign = "+" if sign == "+" else "-"
        symbol = Formula(
            latex=rf"P_{{{latex_sign}}}",
            mathml=f"<msub><mi>P</mi><mo>{sign}</mo></msub>",
        )
        if base_kind.endswith("Upper"):
            symbol = _superscript(symbol, _roman("upper"))
    elif base_kind.startswith("A") and base_kind[1:].isdigit():
        number = int(base_kind[1:])
        symbol = _row(
            Formula(
                latex=r"\gamma_{5}",
                mathml="<msub><mi>γ</mi><mn>5</mn></msub>",
            ),
            Formula(
                latex=rf"\gamma_{{{number}}}",
                mathml=f"<msub><mi>γ</mi><mn>{number}</mn></msub>",
            ),
            latex_separator=r"\,",
        )
    elif base_kind.startswith("T") and base_kind[1:].isdigit():
        indices = base_kind[1:]
        symbol = Formula(
            latex=rf"\sigma_{{{indices}}}",
            mathml=f"<msub><mi>σ</mi><mn>{indices}</mn></msub>",
        )
    else:
        symbol = _subscript(
            Formula(latex=r"\Gamma", mathml="<mi>Γ</mi>"),
            _roman(base_kind),
        )
    if daggered:
        symbol = _superscript(
            _parenthesized(symbol),
            Formula(latex=r"\dagger", mathml="<mo>†</mo>"),
        )
    return symbol


def _render_gamma(matrix: GammaMatrix, context: _FormulaContext) -> Formula:
    indices = _row(context.index(matrix.left), context.index(matrix.right))
    return _subscript(_parenthesized(_gamma_symbol(matrix.kind)), indices)


def _render_field(field: QuarkField, context: _FormulaContext) -> Formula:
    flavor = Formula(
        latex=_escape_latex_text(field.flavor),
        mathml=f"<mi>{html.escape(field.flavor)}</mi>",
    )
    if field.is_anti:
        flavor = Formula(
            latex=rf"\bar{{{flavor.latex}}}",
            mathml=f'<mover accent="true">{flavor.mathml}<mo>¯</mo></mover>',
        )
    indices = _row(context.index(field.color), context.index(field.spin))
    indexed = _subscript(flavor, indices)
    arguments = _row(
        context.index(field.pos),
        _operator_token(","),
        Formula(latex="t", mathml="<mi>t</mi>"),
        latex_separator=r"\,",
    )
    return _row(indexed, _parenthesized(arguments))


def _render_path(hops: Sequence[str]) -> Formula:
    pieces: list[Formula] = []
    for hop in hops:
        sign, axis = hop[0], hop[1:]
        pieces.append(
            Formula(
                latex=rf"{sign}\hat{{{_escape_latex_text(axis)}}}",
                mathml=(
                    f"<mrow><mo>{sign}</mo><mover><mi>{html.escape(axis)}</mi>"
                    "<mo>^</mo></mover></mrow>"
                ),
            )
        )
    if not pieces:
        return _roman("local")
    return _row(*pieces, latex_separator=",")


def _render_apply(application: Apply, context: _FormulaContext) -> Formula:
    operand = _render_expression(application.operand, context)
    operator = application.operator
    if isinstance(operator, CovariantDerivative):
        direction = _render_path((operator.direction,))
        action = _subscript(
            Formula(latex=r"\nabla", mathml="<mi>∇</mi>"),
            direction,
        )
    elif isinstance(operator, ParallelTransport):
        action = _subscript(
            Formula(latex=r"\mathcal{U}", mathml='<mi mathvariant="script">U</mi>'),
            _render_path(operator.path.hops),
        )
    elif isinstance(operator, Shift):
        action = _subscript(
            Formula(latex="T", mathml="<mi>T</mi>"),
            _render_path(operator.path.hops),
        )
    elif isinstance(operator, Projection):
        projector = "LapH" if operator.projector.lower() == "laph" else operator.projector
        action = _subscript(
            Formula(latex=r"\mathcal{P}", mathml='<mi mathvariant="script">P</mi>'),
            _roman(projector),
        )
    else:
        raise PhysicsVisualizationError(
            f"cannot display field operation {type(operator).__name__}"
        )
    return _row(action, _parenthesized(operand), latex_separator=r"\,")


def _render_expression(expression: Expr, context: _FormulaContext) -> Formula:
    if isinstance(expression, Product):
        return _row(
            *(_render_expression(factor, context) for factor in expression.factors),
            latex_separator=r"\,",
        )
    if isinstance(expression, QuarkField):
        return _render_field(expression, context)
    if isinstance(expression, GammaMatrix):
        return _render_gamma(expression, context)
    if isinstance(expression, ColorDelta):
        indices = _row(context.index(expression.left), context.index(expression.right))
        return _subscript(
            Formula(latex=r"\delta", mathml="<mi>δ</mi>"),
            indices,
        )
    if isinstance(expression, ColorEpsilon):
        indices = _row(
            context.index(expression.i),
            context.index(expression.j),
            context.index(expression.k),
        )
        return _subscript(
            Formula(latex=r"\varepsilon", mathml="<mi>ε</mi>"),
            indices,
        )
    if isinstance(expression, GaugeLink):
        colors = _row(
            context.index(expression.start_color),
            context.index(expression.end_color),
        )
        gauge = _subscript(Formula(latex="U", mathml="<mi>U</mi>"), colors)
        arguments = _row(
            context.index(expression.start_pos),
            Formula(latex=r"\to", mathml="<mo>→</mo>"),
            context.index(expression.end_pos),
            _operator_token(","),
            Formula(latex="t", mathml="<mi>t</mi>"),
            latex_separator=r"\,",
        )
        return _row(gauge, _parenthesized(arguments))
    if isinstance(expression, Apply):
        return _render_apply(expression, context)
    if isinstance(expression, Propagator):
        sink = _row(
            context.index(expression.sink_pos),
            context.index(expression.sink_color),
            context.index(expression.sink_spin),
            latex_separator=",",
        )
        source = _row(
            context.index(expression.source_pos),
            context.index(expression.source_color),
            context.index(expression.source_spin),
            latex_separator=",",
        )
        symbol = _subscript(
            Formula(latex="S", mathml="<mi>S</mi>"),
            _roman(expression.flavor),
        )
        arguments = _row(sink, _operator_token(";"), source, latex_separator=r"\,")
        return _row(symbol, _parenthesized(arguments))
    raise PhysicsVisualizationError(
        f"cannot display symbolic expression {type(expression).__name__}"
    )


def _phase_term(position: Formula, momentum: tuple[int, int, int]) -> Formula | None:
    components: list[str] = []
    readable: list[str] = []
    position_name = position.mathml.replace('<mi mathvariant="bold">', "").replace(
        "</mi>", ""
    )
    for axis, coefficient in enumerate(momentum, start=1):
        if coefficient == 0:
            continue
        coordinate = rf"{position.latex}_{{{axis}}}/L_{{{axis}}}"
        if coefficient == 1:
            components.append(coordinate)
        elif coefficient == -1:
            components.append("-" + coordinate)
        else:
            components.append(f"{coefficient}{coordinate}")
        readable.append(f"{coefficient}{position_name}{axis}/L{axis}")
    if not components:
        return None
    scalar = "+".join(components).replace("+-", "-")
    exponent = rf"+2\pi i\left({scalar}\right)"
    readable_exponent = "+2πi(" + " + ".join(readable) + ")"
    return Formula(
        latex=rf"e^{{{exponent}}}",
        mathml=(
            "<msup><mi>e</mi>"
            f"<mtext>{html.escape(readable_exponent)}</mtext></msup>"
        ),
    )


def _render_operator(operator: Operator) -> Formula:
    context = _FormulaContext()
    expression = _render_expression(operator.expr, context)
    positions = context.positions(operator.expr)
    pieces: list[Formula] = []
    if positions:
        summed = _row(*(symbol for _, symbol in positions), latex_separator=",")
        pieces.append(
            _subscript(
                Formula(latex=r"\sum", mathml="<mo>∑</mo>"),
                summed,
            )
        )
    momenta = operator.momenta
    phases = tuple(
        phase
        for index, position in positions
        if (phase := _phase_term(position, momenta.get(index.label, (0, 0, 0))))
        is not None
    )
    pieces.extend(phases)
    pieces.append(expression)
    return _row(*pieces, latex_separator=r"\,")


def _number(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 1e-13:
        return str(int(rounded))
    return f"{value:.12g}"


def _coefficient(value: complex) -> Formula:
    real = 0.0 if abs(value.real) < 1e-15 else value.real
    imag = 0.0 if abs(value.imag) < 1e-15 else value.imag
    if imag == 0.0:
        text = _number(real)
        return Formula(latex=text, mathml=f"<mn>{html.escape(text)}</mn>")
    if real == 0.0:
        magnitude = "" if abs(imag) == 1.0 else _number(abs(imag))
        sign = "-" if imag < 0 else ""
        latex = rf"{sign}{magnitude}\,i"
        return Formula(
            latex=latex,
            mathml=f"<mtext>{html.escape(sign + magnitude + 'i')}</mtext>",
        )
    imag_sign = "+" if imag > 0 else "-"
    imag_magnitude = "" if abs(imag) == 1.0 else _number(abs(imag))
    text = f"{_number(real)}{imag_sign}{imag_magnitude}i"
    return Formula(
        latex=rf"\left({_number(real)}{imag_sign}{imag_magnitude}\,i\right)",
        mathml=f"<mtext>({html.escape(text)})</mtext>",
    )


def _render_combination(combination: OperatorCombination) -> Formula:
    if not combination.terms:
        return Formula(latex="0", mathml="<mn>0</mn>")
    terms: list[Formula] = []
    for index, (coefficient, operator) in enumerate(combination.terms):
        body = _render_operator(operator)
        if coefficient == 1.0 + 0.0j:
            weighted = body
        elif coefficient == -1.0 + 0.0j:
            weighted = _row(_operator_token("-"), body)
        else:
            weighted = _row(
                _coefficient(coefficient),
                _parenthesized(body),
                latex_separator=r"\,",
            )
        if index and not weighted.latex.startswith("-"):
            terms.append(_operator_token("+"))
        terms.append(weighted)
    return _row(*terms, latex_separator=r"\,")


def latex_expression(expression: Expr) -> str:
    """Return conventional LaTeX for one frontend expression."""

    if not isinstance(expression, Expr):
        raise TypeError("expression must be Expr")
    return _render_expression(expression, _FormulaContext()).latex


def latex_operator(operator: Operator | OperatorCombination) -> str:
    """Return the momentum-projected LaTeX right-hand side of an operator."""

    if isinstance(operator, Operator):
        return _render_operator(operator).latex
    if isinstance(operator, OperatorCombination):
        return _render_combination(operator).latex
    raise TypeError("operator must be Operator or OperatorCombination")


def _role_names(arity: int) -> tuple[str, ...]:
    if arity == 1:
        return ("operator",)
    if arity == 2:
        return ("sink", "source")
    if arity == 3:
        return ("sink", "current", "source")
    return tuple(f"insertion {index + 1}" for index in range(arity))


def _role_latex(role: str) -> str:
    return {
        "sink": "snk",
        "source": "src",
        "current": "J",
        "operator": "op",
    }.get(role, role.replace(" ", "_"))


def _operator_symbol(correlator_id: str, role: str) -> Formula:
    base = Formula(
        latex=r"\mathcal{O}",
        mathml='<mi mathvariant="script">O</mi>',
    )
    return _subsup(base, _roman(correlator_id), _roman(_role_latex(role)))


def _at_time(symbol: Formula, role: str) -> Formula:
    time = _subscript(
        Formula(latex="t", mathml="<mi>t</mi>"),
        _roman(_role_latex(role)),
    )
    return _row(symbol, _parenthesized(time))


def _operator_equations(
    correlator_id: str,
    correlator: Correlator,
) -> tuple[tuple[str, Formula], ...]:
    roles = _role_names(correlator.n_operators)
    combinations = correlator.operator_combinations
    equations: list[tuple[str, Formula]] = []
    for index, (role, combination) in enumerate(
        zip(roles, combinations, strict=True)
    ):
        left = _at_time(_operator_symbol(correlator_id, role), role)
        is_dagger_source = (
            role == "source"
            and index == len(combinations) - 1
            and combination == combinations[0].dagger()
        )
        if is_dagger_source:
            sink = _at_time(_operator_symbol(correlator_id, roles[0]), role)
            right = _superscript(
                _parenthesized(sink),
                Formula(latex=r"\dagger", mathml="<mo>†</mo>"),
            )
        else:
            right = _render_combination(combination)
        equations.append(
            (
                role,
                _row(left, _operator_token("="), right, latex_separator=r"\;"),
            )
        )
    return tuple(equations)


def _correlator_formula(correlator_id: str, correlator: Correlator) -> Formula:
    roles = _role_names(correlator.n_operators)
    times = tuple(
        _subscript(
            Formula(latex="t", mathml="<mi>t</mi>"),
            _roman(_role_latex(role)),
        )
        for role in roles
    )
    left = _row(
        _subscript(
            Formula(latex="C", mathml="<mi>C</mi>"),
            _roman(correlator_id),
        ),
        _parenthesized(_row(*times, latex_separator=",")),
    )
    insertions = _row(
        *(
            _at_time(_operator_symbol(correlator_id, role), role)
            for role in roles
        ),
        latex_separator=r"\,",
    )
    expectation = Formula(
        latex=rf"\left\langle {insertions.latex}\right\rangle",
        mathml=(
            '<mrow><mo stretchy="true">⟨</mo>'
            f"{insertions.mathml}"
            '<mo stretchy="true">⟩</mo></mrow>'
        ),
    )
    return _row(left, _operator_token("="), expectation, latex_separator=r"\;")


def latex_correlator(correlator_id: str, correlator: Correlator) -> str:
    """Return LaTeX for the correlator assembled from its ordered insertions."""

    if not isinstance(correlator, Correlator):
        raise TypeError("correlator must be Correlator")
    return _correlator_formula(correlator_id, correlator).latex


def _normalize_collection(collection: OperatorCollection) -> dict[str, Correlator]:
    if isinstance(collection, Correlator):
        return {"C0": collection}
    if isinstance(collection, CorrMatrix):
        return {
            f"C[{row},{column}]": correlator
            for row, column, correlator in collection
        }
    if isinstance(collection, Mapping):
        return dict(collection)
    raise TypeError("collection must be Correlator, CorrMatrix, or a correlator mapping")


def _compact_integers(values: Sequence[int]) -> str:
    ordered = tuple(sorted(set(values)))
    if not ordered:
        return "not specified"
    if len(ordered) == 1:
        return str(ordered[0])
    contiguous = ordered == tuple(range(ordered[0], ordered[-1] + 1))
    if contiguous and len(ordered) > 4:
        return f"{ordered[0]}–{ordered[-1]}"
    if len(ordered) <= 8:
        return ", ".join(str(value) for value in ordered)
    return ", ".join(str(value) for value in ordered[:4]) + f", … ({len(ordered)} total)"


def _time_rows(spec: JobSpec, arity: int) -> tuple[tuple[str, str], ...]:
    roles = _role_names(arity)
    assignments = [
        context.operator_times[arity]
        for context in spec.time_contexts
        if arity in context.operator_times
    ]
    if not assignments:
        return ()
    rows = []
    for index, role in enumerate(roles):
        rows.append(
            (role.title(), _compact_integers([item[index] for item in assignments]))
        )
    rows.append(("Time combinations", str(len(assignments))))
    return tuple(rows)


def _formula_html(formula: Formula) -> str:
    latex = html.escape(formula.latex)
    return f"""
      <div class="formula" role="img" aria-label="{latex}">
        <math display="block">
          <semantics>
            {formula.mathml}
            <annotation encoding="application/x-tex">{latex}</annotation>
          </semantics>
        </math>
      </div>
      <details class="latex-source">
        <summary>LaTeX source</summary>
        <button class="copy-latex" type="button">Copy LaTeX</button>
        <code>{latex}</code>
      </details>
    """


def _definition_list(rows: Sequence[tuple[str, str]]) -> str:
    return "".join(
        f"<div><dt>{html.escape(label)}</dt><dd>{html.escape(value)}</dd></div>"
        for label, value in rows
    )


def _human_spin_basis(value: str) -> str:
    return {
        "DR": "DeGrand–Rossi (DR)",
        "DP": "Dirac–Pauli (DP)",
    }[value]


def _format_duration(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    if seconds < 60.0:
        return f"{seconds:.3f} s"
    minutes, remainder = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)} min {remainder:.1f} s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours} h {minutes} min {remainder:.0f} s"


def _format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(amount) < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(amount)} {unit}"
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    return f"{value} B"  # pragma: no cover


_RESOURCE_LABELS = {
    "propagation": "quark propagation tensors",
    "kernel": "LapH operator tensors",
    "spin": "spin structures",
}


def _resource_label(kind: str, count: int = 1) -> str:
    plural = _RESOURCE_LABELS.get(kind, f"{kind} tensors")
    if count == 1 and plural.endswith("s"):
        return plural[:-1]
    return plural


def _plain_spin_symbol(kind: str) -> str:
    daggered = kind.endswith("_dag")
    base = kind.removesuffix("_dag") if daggered else kind
    symbols = {
        "g5": "γ₅",
        "gamma5": "γ₅",
        "P": "γ₅",
        "g1": "γ₁",
        "gammaX": "γ₁",
        "V1": "γ₁",
        "g2": "γ₂",
        "gammaY": "γ₂",
        "V2": "γ₂",
        "g3": "γ₃",
        "gammaZ": "γ₃",
        "V3": "γ₃",
        "g4": "γ₄",
        "gammaT": "γ₄",
        "V4": "γ₄",
        "Identity": "1",
        "gamma_id": "1",
        "S": "1",
        "C": "C",
        "Cg5": "Cγ₅",
        "Cg5Upper": "(Cγ₅)upper",
        "Pplus": "P₊",
        "Pminus": "P₋",
        "PplusUpper": "(P₊)upper",
    }
    symbol = symbols.get(base, base)
    return f"{symbol}†" if daggered else symbol


def _network_operator_ids(placement: Sequence[object]) -> tuple[int, ...]:
    if not placement:
        return ()
    if placement[0] == "kernel_ops" and len(placement) > 1:
        return tuple(
            value
            for value in placement[1]
            if isinstance(value, int) and not isinstance(value, bool)
        )
    if placement[0] == "spin_op" and len(placement) > 1:
        value = placement[1]
        if isinstance(value, int) and not isinstance(value, bool):
            return (value,)
    return ()


def _network_coefficient(value: complex) -> str:
    def number(part: float) -> str:
        return str(int(part)) if part.is_integer() else f"{part:g}"

    if value.imag == 0.0:
        return number(value.real)
    if value.real == 0.0:
        return f"{number(value.imag)}i"
    sign = "+" if value.imag > 0.0 else "−"
    return f"{number(value.real)} {sign} {number(abs(value.imag))}i"


def _component_network_payload(plan: ConcordPlan, component) -> dict[str, object]:
    operator_views: dict[int, dict[str, object]] = {}

    def operator_view(operator_id: int) -> dict[str, object]:
        return operator_views.setdefault(
            operator_id,
            {
                "id": operator_id,
                "momenta": [],
                "fields": [],
                "spins": [],
                "features": [],
                "open_indices": set(),
            },
        )

    def add_unique(view: dict[str, object], key: str, value: str) -> None:
        values = view[key]
        if value and value not in values:
            values.append(value)

    kernel_endpoints: dict[str, list[dict[str, object]]] = {}
    output_indices = set(component.output_indices)

    for tensor in component.tensors:
        if tensor.kind != "kernel":
            continue
        resource = plan.global_cse.resources[tensor.resource_id]
        operator_ids = _network_operator_ids(tensor.placement)
        payload = resource.payload
        for operator_id in operator_ids:
            view = operator_view(operator_id)
            for momentum in getattr(payload, "momenta", {}).values():
                add_unique(
                    view,
                    "momenta",
                    f"p=({', '.join(str(value) for value in momentum)})",
                )
            for projected in getattr(payload, "projected_fields", ()):
                record = projected.record
                field = f"{record.flavor}̄" if record.is_anti else record.flavor
                view["fields"].append(field)
                for operation in projected.operations:
                    kind = getattr(operation.kind, "value", str(operation.kind))
                    parameters = operation.parameters
                    if "direction" in parameters:
                        kind = f"{kind} {parameters['direction']}"
                    elif "path" in parameters:
                        kind = f"{kind} {' '.join(parameters['path'])}"
                    add_unique(view, "features", kind)
            for color_tensor in getattr(payload, "color_tensors", ()):
                label = (
                    "ε color contraction"
                    if color_tensor.kind == "epsilon"
                    else f"{color_tensor.kind} color contraction"
                )
                add_unique(view, "features", label)
            if getattr(payload, "gauge_tensors", ()):
                add_unique(view, "features", "gauge-linked")
            if len(getattr(payload, "position_labels", ())) > 1:
                add_unique(view, "features", "nonlocal")
        if len(operator_ids) == 1:
            operator_id = operator_ids[0]
            for binding in tensor.bindings:
                if binding.port.startswith("mode_"):
                    kernel_endpoints.setdefault(binding.index, []).append(
                        {"operator_id": operator_id, "port": binding.port}
                    )

    for tensor in component.tensors:
        if tensor.kind != "spin":
            continue
        operator_ids = _network_operator_ids(tensor.placement)
        resource = plan.global_cse.resources[tensor.resource_id]
        for operator_id in operator_ids:
            view = operator_view(operator_id)
            view["spins"].append(_plain_spin_symbol(resource.payload.kind))
            for binding in tensor.bindings:
                if binding.index in output_indices:
                    view["open_indices"].add(binding.index)

    propagators: list[dict[str, object]] = []
    for tensor in component.tensors:
        if tensor.kind != "propagation":
            continue
        resource = plan.global_cse.resources[tensor.resource_id]
        payload = resource.payload
        placement = tensor.placement
        path_operator_ids: list[int] = []
        if placement and placement[0] == "path_ops" and len(placement) == 4:
            candidates = (placement[1], *placement[2], placement[3])
            path_operator_ids = [
                value
                for value in candidates
                if isinstance(value, int) and not isinstance(value, bool)
            ]
        endpoints: list[dict[str, object]] = []
        for binding in tensor.bindings:
            if "mode" not in binding.port:
                continue
            endpoints.extend(kernel_endpoints.get(binding.index, ()))
        unique_endpoints: dict[tuple[int, str], dict[str, object]] = {}
        for endpoint in endpoints:
            key = (endpoint["operator_id"], endpoint["port"])
            unique_endpoints[key] = endpoint
        endpoint_by_operator = {
            endpoint["operator_id"]: endpoint
            for endpoint in unique_endpoints.values()
        }
        route_ids = sorted(
            set(path_operator_ids) | set(endpoint_by_operator), reverse=True
        )
        route = []
        for operator_id in route_ids:
            operator_view(operator_id)
            endpoint = endpoint_by_operator.get(operator_id)
            route.append(
                {
                    "operator_id": operator_id,
                    "port": endpoint["port"] if endpoint is not None else None,
                }
            )
        flavors = list(getattr(payload, "segment_flavors", ()))
        propagators.append(
            {
                "flavor": "→".join(flavors) if flavors else "quark",
                "route": route,
                "generalized": bool(getattr(payload, "insertion_specs", ())),
            }
        )

    if operator_views:
        maximum_operator_id = max(operator_views)
        for operator_id, view in operator_views.items():
            if maximum_operator_id == 0:
                role = "Operator"
            elif operator_id == maximum_operator_id:
                role = "Source operator"
            elif operator_id == 0:
                role = "Sink operator"
            else:
                role = f"Insertion operator {operator_id}"
            view["role"] = role
            view["open_index_count"] = len(view.pop("open_indices"))

    flavor_counts = Counter(propagator["flavor"] for propagator in propagators)
    flavor_seen: Counter[str] = Counter()
    for propagator in propagators:
        flavor = propagator["flavor"]
        flavor_seen[flavor] += 1
        propagator["copy"] = (
            flavor_seen[flavor] if flavor_counts[flavor] > 1 else None
        )

    return {
        "operators": sorted(
            operator_views.values(), key=lambda view: view["id"], reverse=True
        ),
        "propagators": propagators,
        "open_index_count": len(component.output_indices),
    }


def _network_payload(
    plan: ConcordPlan,
    operator_labels: Mapping[str, Sequence[str]],
) -> str:
    payload = {
        "correlators": [
            {
                "id": correlator.correlator_id,
                "operator_labels": list(
                    operator_labels.get(correlator.correlator_id, ())
                ),
                "terms": [
                    {
                        "index": term.term_index,
                        "coefficient": _network_coefficient(term.coefficient),
                        "components": [
                            use.computation_id for use in term.components
                        ],
                    }
                    for term in correlator.terms
                ],
            }
            for correlator in plan.global_cse.correlators
        ],
        "components": {
            component.computation_id: _component_network_payload(plan, component)
            for component in plan.global_cse.components.values()
        },
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).replace("</", "<\\/")


_NETWORK_SCRIPT = r"""
<script>
(() => {
  const payloadElement = document.getElementById("tensor-network-data");
  const channelSelect = document.getElementById("network-channel");
  const termSelect = document.getElementById("network-term");
  const canvas = document.getElementById("tensor-network-canvas");
  const caption = document.getElementById("network-caption");
  if (!payloadElement || !channelSelect || !termSelect || !canvas || !caption) return;

  const payload = JSON.parse(payloadElement.textContent);
  const svgNamespace = "http://www.w3.org/2000/svg";
  const flavorClasses = {u: "flavor-u", d: "flavor-d", s: "flavor-s", c: "flavor-c", b: "flavor-b"};
  const flavorColors = {u: "#315b73", d: "#a14d36", s: "#557a55", c: "#8c651f", b: "#493d70", other: "#315b73"};

  function svgElement(name, attributes = {}, text = null) {
    const element = document.createElementNS(svgNamespace, name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    if (text !== null) element.textContent = text;
    return element;
  }

  function svgText(group, x, y, text, className, anchor = "start") {
    group.appendChild(svgElement("text", {
      x, y, class: className, "text-anchor": anchor,
    }, text));
  }

  function clipped(values, fallback, limit = 3) {
    if (!values?.length) return fallback;
    const shown = values.slice(0, limit).join(", ");
    return values.length > limit ? `${shown}, …` : shown;
  }

  function drawComponent(componentId, component, lane, totalLanes, operatorLabels) {
    const laneHeight = 350;
    const width = 960;
    const laneTop = lane * laneHeight;
    const centerY = laneTop + 184;
    const nodeRadiusX = 68;
    const nodeRadiusY = 37;
    const operators = component.operators;
    const group = svgElement("g", {class: "tensor-component"});
    svgText(group, 22, laneTop + 28,
      totalLanes > 1 ? `Disconnected contribution ${lane + 1}` : "Quark-flow topology",
      "component-title");

    const definitions = svgElement("defs");
    Object.entries(flavorColors).forEach(([flavor, color]) => {
      const marker = svgElement("marker", {
        id: `quark-arrow-${lane}-${flavor}`,
        markerWidth: 9,
        markerHeight: 9,
        refX: 8,
        refY: 4.5,
        orient: "auto",
        markerUnits: "userSpaceOnUse",
      });
      marker.appendChild(svgElement("path", {
        d: "M 0 0 L 9 4.5 L 0 9 z",
        fill: color,
      }));
      definitions.appendChild(marker);
    });
    group.appendChild(definitions);

    if (!operators.length) {
      svgText(group, width / 2, centerY, "Scalar factor with no operator-local tensor", "empty-network", "middle");
      canvas.appendChild(group);
      return;
    }

    const left = 125;
    const right = width - 125;
    const operatorPositions = new Map();
    operators.forEach((operator, index) => {
      const fraction = operators.length === 1 ? 0.5 : index / (operators.length - 1);
      operatorPositions.set(operator.id, {
        x: left + fraction * (right - left),
        y: centerY,
      });
    });

    const maximumPorts = new Map();
    component.propagators.forEach(propagator => {
      propagator.route.forEach((endpoint, fallbackIndex) => {
        const match = /_(\d+)$/.exec(endpoint.port || "");
        const port = match ? Number(match[1]) : fallbackIndex;
        maximumPorts.set(endpoint.operator_id, Math.max(maximumPorts.get(endpoint.operator_id) ?? 0, port));
      });
    });

    function endpointY(endpoint, lineIndex) {
      const match = /_(\d+)$/.exec(endpoint.port || "");
      const port = match ? Number(match[1]) : lineIndex;
      const maximum = maximumPorts.get(endpoint.operator_id) ?? 0;
      const spacing = Math.min(28, (nodeRadiusY * 1.7) / Math.max(maximum, 1));
      return centerY + (port - maximum / 2) * spacing;
    }

    function ellipseEdgeX(position, y, side) {
      const verticalOffset = y - position.y;
      const ellipseWidth = nodeRadiusX * Math.sqrt(
        Math.max(0, 1 - (verticalOffset / nodeRadiusY) ** 2),
      );
      return position.x + side * ellipseWidth;
    }

    function endpointX(endpoint, lineIndex, side) {
      return ellipseEdgeX(
        operatorPositions.get(endpoint.operator_id),
        endpointY(endpoint, lineIndex),
        side,
      );
    }

    component.propagators.forEach((propagator, lineIndex) => {
      const route = propagator.route
        .map(endpoint => ({endpoint, position: operatorPositions.get(endpoint.operator_id)}))
        .filter(item => item.position);
      if (!route.length) return;
      const baseFlavor = propagator.flavor.split("→")[0].trim();
      const flavorClass = flavorClasses[baseFlavor] || "flavor-other";
      const markerFlavor = flavorColors[baseFlavor] ? baseFlavor : "other";
      const markerEnd = `url(#quark-arrow-${lane}-${markerFlavor})`;
      if (route.length === 1) {
        const point = route[0].position;
        const y = endpointY(route[0].endpoint, lineIndex);
        const direction = point.x > width / 2 ? -1 : 1;
        const startX = endpointX(route[0].endpoint, lineIndex, direction);
        const endY = y <= centerY
          ? Math.min(centerY + nodeRadiusY * 0.85, y + 22)
          : Math.max(centerY - nodeRadiusY * 0.85, y - 22);
        const endX = ellipseEdgeX(point, endY, direction);
        group.appendChild(svgElement("path", {
          d: `M ${startX} ${y} C ${startX + direction * 88} ${y - 76}, ${startX + direction * 88} ${y + 76}, ${endX} ${endY}`,
          class: `quark-line ${flavorClass}`,
          fill: "none",
          "marker-end": markerEnd,
        }));
      } else {
        for (let segmentIndex = 0; segmentIndex < route.length - 1; segmentIndex += 1) {
          const start = route[segmentIndex];
          const end = route[segmentIndex + 1];
          const attributes = {
            d: `M ${endpointX(start.endpoint, lineIndex, 1)} ${endpointY(start.endpoint, lineIndex)} L ${endpointX(end.endpoint, lineIndex, -1)} ${endpointY(end.endpoint, lineIndex)}`,
            class: `quark-line ${flavorClass}`,
            fill: "none",
          };
          if (segmentIndex === route.length - 2) {
            attributes["marker-end"] = markerEnd;
          }
          group.appendChild(svgElement("path", attributes));
        }
        const first = route[0];
        const second = route[1];
        const labelX = (
          endpointX(first.endpoint, lineIndex, 1)
          + endpointX(second.endpoint, lineIndex, -1)
        ) / 2;
        const labelY = (
          endpointY(first.endpoint, lineIndex)
          + endpointY(second.endpoint, lineIndex)
        ) / 2 - 8;
        const copy = propagator.copy ? ` ${propagator.copy}` : "";
        svgText(group, labelX, labelY,
          `${propagator.flavor}${copy}${propagator.generalized ? " with insertion" : ""}`,
          `quark-label ${flavorClass}`, "middle");
      }
    });

    operators.forEach(operator => {
      const position = operatorPositions.get(operator.id);
      const roleClass = operator.role.startsWith("Source")
        ? "source-operator"
        : operator.role.startsWith("Sink")
          ? "sink-operator"
          : "insertion-operator";
      const node = svgElement("g", {class: `operator-node ${roleClass}`});
      const label = operatorLabels[operator.id] || operator.role;
      node.appendChild(svgElement("ellipse", {
        cx: position.x,
        cy: position.y,
        rx: nodeRadiusX,
        ry: 37,
      }));
      svgText(node, position.x, position.y - 48, operator.role, "operator-role", "middle");
      const shortLabel = label.length > 20 ? `${label.slice(0, 18)}…` : label;
      svgText(node, position.x, position.y + 5, shortLabel, "operator-label", "middle");
      svgText(node, position.x, position.y + 58,
        clipped(operator.fields, "quark fields"), "operator-feature", "middle");
      node.appendChild(svgElement("title", {}, [
        label,
        operator.role,
        `Momentum: ${operator.momenta.join(", ") || "none"}`,
        `Fields: ${operator.fields.join(", ") || "implicit"}`,
        `Spin factors: ${operator.spins.join(", ") || "identity"}`,
        `Features: ${operator.features.join(", ") || "local LapH structure"}`,
      ].join("\n")));

      if (operator.open_index_count) {
        const outward = operator.id === operators[0].id ? -1 : 1;
        for (let index = 0; index < operator.open_index_count; index += 1) {
          const y = position.y + (index - (operator.open_index_count - 1) / 2) * 22;
          const x1 = position.x + outward * nodeRadiusX;
          const x2 = x1 + outward * 42;
          node.appendChild(svgElement("line", {x1, y1: y, x2, y2: y, class: "open-spin-line"}));
        }
        const labelX = position.x + outward * (nodeRadiusX + 21);
        svgText(node, labelX, position.y - 18,
          operator.open_index_count === 1 ? "open spin" : `${operator.open_index_count} open spins`,
          "open-spin-label", "middle");
      }
      group.appendChild(node);
    });

    canvas.appendChild(group);
  }

  function draw() {
    const correlator = payload.correlators[channelSelect.selectedIndex];
    const term = correlator?.terms[termSelect.selectedIndex];
    canvas.replaceChildren();
    if (!term) {
      canvas.setAttribute("viewBox", "0 0 960 160");
      canvas.appendChild(svgElement("text", {x: 480, y: 80, "text-anchor": "middle", class: "component-title"}, "This correlator has no surviving Wick terms."));
      caption.textContent = correlator ? correlator.id : "No correlator selected";
      return;
    }
    const laneHeight = 350;
    const componentIds = term.components;
    canvas.setAttribute("viewBox", `0 0 960 ${Math.max(1, componentIds.length) * laneHeight}`);
    if (componentIds.length) {
      componentIds.forEach((componentId, lane) => drawComponent(
        componentId,
        payload.components[componentId],
        lane,
        componentIds.length,
        correlator.operator_labels || [],
      ));
    } else {
      canvas.appendChild(svgElement("text", {x: 480, y: 215, "text-anchor": "middle", class: "component-title"}, "Scalar identity contribution"));
    }
    const lines = componentIds.flatMap(componentId => payload.components[componentId].propagators);
    const flavors = lines.map(line => line.flavor).join(", ") || "none";
    const openCount = componentIds.reduce((sum, componentId) => sum + payload.components[componentId].open_index_count, 0);
    const openness = openCount
      ? `${openCount} open spin ${openCount === 1 ? "index" : "indices"}`
      : "closed scalar contraction";
    const pairingNote = correlator.terms.length > 1
      ? " Terms with the same quark-line topology can still differ in their internal spin-index pairing."
      : "";
    caption.textContent = `${correlator.id}: Wick term ${term.index + 1} has coefficient ${term.coefficient}; quark lines: ${flavors}; ${openness}.${pairingNote}`;
  }

  function updateTerms() {
    const correlator = payload.correlators[channelSelect.selectedIndex];
    termSelect.replaceChildren();
    (correlator?.terms || []).forEach(term => {
      const option = document.createElement("option");
      option.textContent = `Wick term ${term.index + 1} — coefficient ${term.coefficient}`;
      termSelect.appendChild(option);
    });
    draw();
  }

  channelSelect.addEventListener("change", updateTerms);
  termSelect.addEventListener("change", draw);
  updateTerms();
})();
</script>
"""


def _network_html(
    plan: ConcordPlan | None,
    operator_labels: Mapping[str, Sequence[str]],
) -> str:
    if plan is None:
        return """
          <section class="panel empty-state">
            <h2>Tensor network unavailable</h2>
            <p>Compile the job plan first, then regenerate this report. The physics formulae above do not require a compiled plan.</p>
          </section>
        """

    dag = plan.global_cse
    resource_counts = Counter(resource.kind for resource in dag.resources.values())
    tensor_occurrences = sum(
        len(component.tensors) for component in dag.components.values()
    )
    term_count = sum(len(correlator.terms) for correlator in dag.correlators)
    network_payload = _network_payload(plan, operator_labels)
    channel_options = "".join(
        f"<option>{html.escape(correlator.correlator_id)}</option>"
        for correlator in dag.correlators
    )
    input_lines = "".join(
        f"<span>{count} {html.escape(_resource_label(kind, count))}</span>"
        for kind, count in sorted(resource_counts.items())
    ) or "<span>No primitive tensors</span>"
    correlator_details: list[str] = []
    for correlator in dag.correlators:
        terms = "".join(
            f"""
            <li>
              <strong>Wick term {term.term_index + 1}</strong>
              <span>coefficient {html.escape(str(term.coefficient))}</span>
              <span>{len(term.components)} connected component(s):
                {', '.join(html.escape(use.computation_id) for use in term.components) or 'scalar identity'}
              </span>
            </li>
            """
            for term in correlator.terms
        )
        correlator_details.append(
            f"""
            <details class="network-details">
              <summary>{html.escape(correlator.correlator_id)} — {len(correlator.terms)} Wick term(s)</summary>
              <ol>{terms}</ol>
            </details>
            """
        )

    component_details: list[str] = []
    for component in dag.components.values():
        tensors = "".join(
            f"""
            <li>
              <span class="tensor-kind">{html.escape(_resource_label(tensor.kind))}</span>
              <code>{html.escape(tensor.resource_id)}</code>
              <span class="indices">indices: {html.escape(', '.join(binding.index for binding in tensor.bindings) or 'none')}</span>
            </li>
            """
            for tensor in component.tensors
        )
        component_details.append(
            f"""
            <details class="network-details component-detail">
              <summary>{html.escape(component.computation_id)} — {len(component.tensors)} tensor occurrence(s)</summary>
              <ul>{tensors}</ul>
              <p class="note">Open indices: {html.escape(', '.join(component.output_indices) or 'none (scalar)')}</p>
            </details>
            """
        )

    return f"""
      <section class="panel">
        <p class="eyebrow">Contraction structure</p>
        <h2>How the requested correlators are assembled</h2>
        <p class="intro">This is the compiled tensor network after exact algebraic reuse. Repeated tensors and connected contractions appear once in the plan and may feed several Wick terms or correlators.</p>
        <div class="network-flow" aria-label="Tensor-network overview">
          <div class="network-node inputs"><strong>Input tensors</strong>{input_lines}</div>
          <span class="network-arrow" aria-hidden="true">→</span>
          <div class="network-node"><strong>{len(dag.components)}</strong><span>connected contractions</span><small>{tensor_occurrences} tensor occurrences</small></div>
          <span class="network-arrow" aria-hidden="true">→</span>
          <div class="network-node"><strong>{term_count}</strong><span>Wick terms</span></div>
          <span class="network-arrow" aria-hidden="true">→</span>
          <div class="network-node output"><strong>{len(dag.correlators)}</strong><span>correlators</span></div>
        </div>
      </section>
      <section class="panel">
        <p class="eyebrow">Interactive diagram</p>
        <h2>Quark-flow diagram</h2>
        <p class="intro">Choose a correlator and Wick term to see its quark-flow picture. Each arrow is a quark propagator flowing from source to sink; closed lines are loops. Optional operator labels make multi-hadron diagrams easier to read. The exact Wick coefficient and contraction details remain available below.</p>
        <div class="network-controls">
          <label for="network-channel">
            <span>Correlator</span>
            <select id="network-channel">{channel_options}</select>
          </label>
          <label for="network-term">
            <span>Wick term</span>
            <select id="network-term"></select>
          </label>
        </div>
        <div class="network-legend" aria-label="Tensor-network legend">
          <span><i class="legend-node source"></i>Source</span>
          <span><i class="legend-node sink"></i>Sink</span>
          <span><i class="legend-line quark"></i>Quark flow (source → sink)</span>
          <span><i class="legend-line open-spin"></i>Open spin index</span>
        </div>
        <div class="tensor-canvas-shell">
          <svg id="tensor-network-canvas" role="img" aria-labelledby="network-caption"></svg>
        </div>
        <p id="network-caption" class="note"></p>
        <script id="tensor-network-data" type="application/json">{network_payload}</script>
      </section>
      <details class="panel network-audit">
        <summary>Technical contraction details</summary>
        <div class="network-audit-body">
          <h2>Correlator assembly</h2>
          <p class="note">For exact auditing, expand a channel to see every Wick term and the connected components it uses.</p>
          {''.join(correlator_details)}
          <h2>Connected component definitions</h2>
          <p class="note">These definitions expose the complete primitive-tensor content and contraction indices without displaying compiler buffer names.</p>
          {''.join(component_details)}
        </div>
      </details>
      {_NETWORK_SCRIPT}
    """


_STAGE_PRESENTATION = {
    "result_management": (
        "Opening and resuming results",
        "Result-file validation and completion-grid checks",
    ),
    "runtime_planning": (
        "Choosing the runtime strategy",
        "Device, memory, batching, cache, and prefetch decisions",
    ),
    "data_preparation": (
        "Preparing input data",
        "Constructing data providers and staging requests",
    ),
    "input_wait": (
        "Waiting for prefetched input",
        "Wall time blocked on asynchronous input becoming ready",
    ),
    "numerical_execution": (
        "Executing tensor operations",
        "CPU execution or GPU work submission at the numerical executor boundary",
    ),
    "output_writing": (
        "Finalizing and writing correlators",
        "Waiting for results, device-to-host transfer, HDF5 writes, and durable completion markers",
    ),
    "other": (
        "Orchestration and cleanup",
        "Small Python bookkeeping, resource cleanup, and uninstrumented overhead",
    ),
}


def _hardware_html(record: JobRunRecord) -> str:
    hardware = record.hardware
    if hardware is None:
        return """
          <section class="panel hardware-panel">
            <p class="eyebrow">Hardware resources</p>
            <h2>Resources used for this run</h2>
            <p class="note">Hardware identity is unavailable because this invocation performed no numerical work or the record predates hardware capture.</p>
          </section>
        """

    compute = hardware.compute_name
    if hardware.compute_device != "not-required":
        compute = f"{compute} ({hardware.compute_device})"
    rows: list[tuple[str, str]] = [
        ("Compute hardware", compute),
        ("CPU host", hardware.host_name),
    ]
    if hardware.logical_cpus is not None:
        rows.append(("Logical CPUs visible", str(hardware.logical_cpus)))
    if hardware.host_memory_bytes is not None:
        rows.append(("Host memory visible", _format_bytes(hardware.host_memory_bytes)))
    if (
        hardware.compute_memory_bytes is not None
        and hardware.compute_device.startswith("cuda:")
    ):
        rows.append(
            ("Compute-device memory", _format_bytes(hardware.compute_memory_bytes))
        )
    if hardware.torch_num_threads is not None:
        thread_value = str(hardware.torch_num_threads)
        if hardware.torch_num_interop_threads is not None:
            thread_value += f" compute, {hardware.torch_num_interop_threads} inter-op"
        rows.append(("PyTorch CPU threads", thread_value))
    return f"""
      <section class="panel hardware-panel">
        <p class="eyebrow">Hardware resources</p>
        <h2>Resources used for this run</h2>
        <dl class="plain-list">{_definition_list(rows)}</dl>
      </section>
    """


def _runtime_html(record: JobRunRecord | None) -> str:
    if record is None:
        return """
          <section class="panel empty-state">
            <h2>No execution record attached</h2>
            <p>This pre-run report describes the physics and compiled network. Generate a report from <code>concord run --report</code> to include measured wall time.</p>
          </section>
        """

    total = record.timing.execution_wall_seconds
    rows: list[str] = []
    segments: list[str] = []
    for index, (key, seconds) in enumerate(record.timing.stages):
        label, description = _STAGE_PRESENTATION[key]
        share = 0.0 if total == 0.0 else 100.0 * seconds / total
        rows.append(
            f"""
            <div class="timing-row">
              <div class="timing-label">
                <span class="stage-key stage-{index}" aria-hidden="true"></span>
                <div><strong>{html.escape(label)}</strong><small>{html.escape(description)}</small></div>
              </div>
              <div class="timing-value"><strong>{html.escape(_format_duration(seconds))}</strong><span>{share:.1f}%</span></div>
            </div>
            """
        )
        if seconds > 0.0:
            segments.append(
                f'<span class="segment stage-{index}" style="width:{share:.8f}%" title="{html.escape(label)}: {_format_duration(seconds)}"></span>'
            )

    dominant_key, dominant_seconds = max(
        record.timing.stages,
        key=lambda item: item[1],
    )
    dominant_label = _STAGE_PRESENTATION[dominant_key][0]
    dominant_share = 0.0 if total == 0.0 else 100.0 * dominant_seconds / total
    dominant_summary = (
        "No measurable execution time was recorded."
        if total == 0.0
        else (
            f"Most observed wall time was spent on {dominant_label.lower()}: "
            f"{_format_duration(dominant_seconds)} ({dominant_share:.1f}%)."
        )
    )
    status = "Complete" if record.all_complete else "Incomplete"
    kernel_cache_html = ""
    if record.kernel_cache_enabled:
        cache_rows = (
            ("Reused from earlier runs", str(record.kernel_cache_hits)),
            ("Computed afresh", str(record.kernel_cache_misses)),
            ("Cache writes", str(record.kernel_cache_writes)),
        )
        if record.kernel_cache_invalid_entries:
            cache_rows += (
                ("Invalid saved entries ignored", str(record.kernel_cache_invalid_entries)),
            )
        if record.kernel_cache_write_failures:
            cache_rows += (
                ("Entries not saved", str(record.kernel_cache_write_failures)),
            )
        kernel_cache_html = f"""
          <section class="panel">
            <p class="eyebrow">Reusable kernels</p>
            <h2>Preparation work reused or saved</h2>
            <dl class="plain-list">{_definition_list(cache_rows)}</dl>
          </section>
        """
    return f"""
      <section class="overview runtime-overview">
        <div class="summary-card"><strong>{html.escape(_format_duration(total))}</strong><span>Execution wall time</span></div>
        <div class="summary-card"><strong>{record.completed_work_units}/{record.total_work_units}</strong><span>Completed work units</span></div>
        <div class="summary-card"><strong>{html.escape(status)}</strong><span>Run status</span></div>
        <div class="summary-card"><strong>{len(record.correlator_ids)}</strong><span>Correlator channels</span></div>
      </section>
      {_hardware_html(record)}
      {kernel_cache_html}
      <section class="panel">
        <p class="eyebrow">Measured execution</p>
        <h2>Where the wall time went</h2>
        <p class="intro">The colored bar and rows use exclusive host-observed categories that sum exactly to the execution wall time. Plan preparation is shown separately because a saved plan is reused across configurations.</p>
        <p class="runtime-insight">{html.escape(dominant_summary)}</p>
        <div class="stacked-time" aria-label="Wall-time decomposition">{''.join(segments)}</div>
        <div class="timing-table">{''.join(rows)}</div>
        <div class="one-time-cost"><span>One-time plan preparation</span><strong>{html.escape(_format_duration(record.plan_preparation_seconds))}</strong><small>{html.escape(record.plan_action)}</small></div>
        <p class="note record-note">These timings are measured at host boundaries. On accelerator runs, device work may be asynchronous; use a separate device profiler for kernel-level timing.</p>
      </section>
    """


def _numerical_record_html(spec: JobSpec, record: JobRunRecord | None) -> str:
    if record is None:
        status_rows = (
            ("Execution status", "not run or no record supplied"),
            ("Numerical precision", spec.compile.scalar_type),
            ("Spin basis", _human_spin_basis(spec.compile.spin_basis)),
            ("Requested work units", str(len(spec.configurations) * len(spec.time_contexts))),
        )
        note = "No statement about numerical completion can be made from a pre-run report."
        advanced = ""
    else:
        status_rows = (
            ("Execution status", "complete" if record.all_complete else "incomplete"),
            ("Numerical precision", spec.compile.scalar_type),
            ("Spin basis", _human_spin_basis(spec.compile.spin_basis)),
            ("Completed work units", f"{record.completed_work_units} of {record.total_work_units}"),
            ("Correlator channels", str(len(record.correlator_ids))),
        )
        note = "Completion means every requested output slot has a durable completion marker. It does not by itself establish physical correctness."
        peak_tile = (
            _format_bytes(record.peak_tile_bytes)
            if record.peak_tile_bytes is not None
            else "not required"
        )
        advanced_rows = (
            ("Result file", str(record.output_file)),
            ("Plan fingerprint", record.plan_sha256),
            ("Contraction strategy", record.contraction_path),
            ("Strategy selection", record.contraction_selection_source),
            ("Time slices processed together", str(record.time_batch_size)),
            ("Batch selection", record.time_batch_selection_source),
            ("Numerical batches", str(record.executed_batches)),
            ("Compute device", record.selected_device),
            ("Tensor storage", record.storage_device),
            ("Execution mode", record.execution_mode),
            ("Estimated working memory", _format_bytes(record.known_working_set_bytes)),
            ("Largest tiled operation", peak_tile),
            ("Resident tensor-pool peak", _format_bytes(record.resident_pool_peak_bytes)),
            ("Cached tensor requests", f"{record.primitive_cache_hits} hits, {record.primitive_cache_misses} misses"),
            ("HDF5 file use", f"{record.hdf5_file_opens} opens, {record.hdf5_file_reuses} reuses"),
            ("Prefetched propagation data", _format_bytes(record.propagation_prefetch_bytes)),
        )
        advanced = f"""
          <details class="advanced-panel">
            <summary>Advanced execution details</summary>
            <dl class="plain-list">{_definition_list(advanced_rows)}</dl>
          </details>
        """
    return f"""
      <section class="panel">
        <p class="eyebrow">Numerical record</p>
        <h2>What was requested and completed</h2>
        <dl class="plain-list">{_definition_list(status_rows)}</dl>
        <p class="note record-note">{html.escape(note)}</p>
        {advanced}
      </section>
    """


def render_job_visualization(
    spec: JobSpec,
    collection: OperatorCollection,
    *,
    plan: ConcordPlan | None = None,
    run_record: JobRunRecord | None = None,
) -> str:
    """Render one self-contained physicist-facing calculation report."""

    if not isinstance(spec, JobSpec):
        raise TypeError("spec must be JobSpec")
    if plan is not None and not isinstance(plan, ConcordPlan):
        raise TypeError("plan must be ConcordPlan or None")
    if run_record is not None and not isinstance(run_record, JobRunRecord):
        raise TypeError("run_record must be JobRunRecord or None")
    if plan is not None and run_record is not None:
        if tuple(plan.correlator_ids) != run_record.correlator_ids:
            raise PhysicsVisualizationError(
                "run record correlators differ from the supplied compiled plan"
            )
    if run_record is not None and run_record.job_name != spec.job.name:
        raise PhysicsVisualizationError("run record belongs to a different job")
    correlators = _normalize_collection(collection)
    if not correlators:
        raise PhysicsVisualizationError("at least one correlator is required")
    unknown_label_channels = set(spec.visualization.operator_labels) - set(correlators)
    if unknown_label_channels:
        raise PhysicsVisualizationError(
            "visualization.operator_labels names unknown correlator channel(s): "
            + ", ".join(sorted(unknown_label_channels))
        )
    for correlator_id, labels in spec.visualization.operator_labels.items():
        expected = correlators[correlator_id].n_operators
        if len(labels) != expected:
            raise PhysicsVisualizationError(
                f"visualization.operator_labels[{correlator_id!r}] must contain "
                f"one label for each of its {expected} operator insertions"
            )

    lattice = (*spec.lattice.spatial_extent, spec.lattice.temporal_extent)
    overview = (
        ("Correlator channels", str(len(correlators))),
        ("Gauge configurations", str(len(spec.configurations))),
        ("Time combinations", str(len(spec.time_contexts))),
        ("Lattice", " × ".join(str(extent) for extent in lattice)),
    )
    setup = (
        ("LapH modes", str(spec.compile.mode_size)),
        ("Spin components", str(spec.compile.spin_size)),
        ("Spin basis", _human_spin_basis(spec.compile.spin_basis)),
        ("Gauge links", "included" if spec.data.gauge_links is not None else "not used"),
    )

    channel_links = "".join(
        f'<a href="#channel-{index}">{html.escape(correlator_id)}</a>'
        for index, correlator_id in enumerate(correlators)
    )
    channel_sections: list[str] = []
    for channel_index, (correlator_id, correlator) in enumerate(correlators.items()):
        arity = correlator.n_operators
        kind = {
            1: "one-point function",
            2: "two-point function",
            3: "three-point function",
        }.get(arity, f"{arity}-point function")
        equations = _operator_equations(correlator_id, correlator)
        operator_cards = "".join(
            f"""
            <article class="operator-card">
              <h3>{html.escape(role.title())} operator</h3>
              {_formula_html(equation)}
            </article>
            """
            for role, equation in equations
        )
        time_rows = _time_rows(spec, arity)
        time_block = (
            f'<dl class="plain-list">{_definition_list(time_rows)}</dl>'
            if time_rows
            else '<p class="note">No time assignment was supplied for this correlator arity.</p>'
        )
        search_text = " ".join(
            (correlator_id, kind, *(equation.latex for _, equation in equations))
        ).lower()
        channel_sections.append(
            f"""
            <section class="channel" id="channel-{channel_index}" data-channel-search="{html.escape(search_text)}">
              <div class="channel-heading">
                <div>
                  <p class="eyebrow">{html.escape(kind)}</p>
                  <h2>{html.escape(correlator_id)}</h2>
                </div>
                <a class="back" href="#top">Back to summary</a>
              </div>
              <h3 class="section-label">Resulting correlator</h3>
              {_formula_html(_correlator_formula(correlator_id, correlator))}
              <div class="operator-grid">{operator_cards}</div>
              <div class="times">
                <h3>Times included in this job</h3>
                {time_block}
              </div>
            </section>
            """
        )

    description = (
        f'<p class="description">{html.escape(spec.job.description)}</p>'
        if spec.job.description
        else ""
    )
    title = html.escape(spec.job.name)
    overview_cards = "".join(
        f'<div class="summary-card"><strong>{html.escape(value)}</strong>'
        f'<span>{html.escape(label)}</span></div>'
        for label, value in overview
    )
    network_html = _network_html(plan, spec.visualization.operator_labels)
    runtime_html = _runtime_html(run_record)
    numerical_html = _numerical_record_html(spec, run_record)
    report_intro = (
        "This report joins the symbolic physics, compiled tensor network, and measured execution record."
        if run_record is not None
        else "This pre-run report shows the symbolic physics and any available compiled tensor network."
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ConCorD calculation report — {title}</title>
  <style>
    :root {{ color-scheme: light; --ink: #17212b; --muted: #596675; --paper: #f5f1e8; --card: #fffdf8; --accent: #7b2d26; --accent-soft: #f2ddd7; --line: #d8d0c1; --formula: #eef3ef; --blue: #315b73; --green: #557a55; --gold: #a47128; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--paper); color: var(--ink); font-family: Georgia, "Times New Roman", serif; line-height: 1.55; }}
    main {{ width: min(1120px, calc(100% - 32px)); margin: 0 auto; padding: 42px 0 80px; }}
    h1, h2, h3, p {{ margin-top: 0; }}
    h1 {{ font-size: clamp(2rem, 5vw, 4rem); line-height: 1.05; margin-bottom: 16px; }}
    h2 {{ font-size: 1.8rem; margin-bottom: 0; }}
    h3 {{ font-size: 1.05rem; }}
    .eyebrow {{ color: var(--accent); font-family: system-ui, sans-serif; font-size: .76rem; font-weight: 750; letter-spacing: .12em; margin-bottom: 8px; text-transform: uppercase; }}
    .description, .intro {{ color: var(--muted); font-size: 1.08rem; max-width: 760px; }}
    .overview {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 30px 0; }}
    .summary-card, .panel, .channel, .operator-card {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px; }}
    .summary-card {{ padding: 18px; }}
    .summary-card strong {{ display: block; color: var(--accent); font-family: system-ui, sans-serif; font-size: 1.4rem; }}
    .summary-card span {{ color: var(--muted); font-family: system-ui, sans-serif; font-size: .88rem; }}
    .panel {{ margin: 18px 0; padding: 22px; }}
    .plain-list {{ margin: 0; }}
    .plain-list > div {{ display: grid; grid-template-columns: minmax(150px, 1fr) 2fr; gap: 18px; padding: 8px 0; border-bottom: 1px solid var(--line); }}
    .plain-list > div:last-child {{ border-bottom: 0; }}
    dt {{ color: var(--muted); }}
    dd {{ margin: 0; font-weight: 600; }}
    .channel-index {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .channel-index a, .back {{ color: var(--accent); font-family: system-ui, sans-serif; }}
    .channel-index a {{ background: var(--accent-soft); border-radius: 999px; padding: 6px 11px; text-decoration: none; }}
    .view-tabs {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 24px 0 30px; position: sticky; top: 8px; z-index: 10; }}
    .view-tabs a {{ background: #ece6da; border: 1px solid var(--line); border-radius: 999px; color: var(--ink); font-family: system-ui, sans-serif; font-size: .9rem; font-weight: 700; padding: 9px 14px; text-decoration: none; }}
    .view-tabs a.active {{ background: var(--accent); border-color: var(--accent); color: white; }}
    .js .report-view {{ display: none; }}
    .js .report-view.active {{ display: block; }}
    .channel {{ margin-top: 30px; padding: clamp(18px, 4vw, 34px); }}
    .channel-heading {{ display: flex; align-items: start; justify-content: space-between; gap: 20px; margin-bottom: 24px; }}
    .back {{ font-size: .86rem; white-space: nowrap; }}
    .section-label {{ margin: 0 0 10px; }}
    .formula {{ background: var(--formula); border-left: 4px solid var(--accent); border-radius: 8px; overflow-x: auto; padding: 17px; }}
    math {{ font-size: 1.15rem; min-width: max-content; }}
    .latex-source {{ margin: 8px 0 20px; color: var(--muted); font-family: system-ui, sans-serif; font-size: .82rem; }}
    .latex-source summary {{ cursor: pointer; }}
    .latex-source code {{ display: block; margin-top: 8px; overflow-x: auto; padding: 10px; background: #f2eee5; border-radius: 7px; color: var(--ink); white-space: pre-wrap; }}
    .copy-latex {{ background: var(--card); border: 1px solid var(--line); border-radius: 6px; color: var(--accent); cursor: pointer; float: right; font: inherit; margin: 6px 0; padding: 5px 9px; }}
    .operator-grid {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(min(100%, 430px), 1fr)); margin-top: 24px; }}
    .operator-card {{ padding: 18px; }}
    .operator-card h3 {{ color: var(--accent); font-family: system-ui, sans-serif; }}
    .times {{ margin-top: 22px; }}
    .note {{ color: var(--muted); }}
    .channel-search {{ border: 1px solid var(--line); border-radius: 8px; font: inherit; margin-top: 14px; max-width: 420px; padding: 10px 12px; width: 100%; }}
    .empty-state {{ border-style: dashed; }}
    .network-flow {{ align-items: stretch; display: grid; gap: 10px; grid-template-columns: minmax(170px, 1.5fr) auto minmax(150px, 1fr) auto minmax(130px, .8fr) auto minmax(130px, .8fr); margin-top: 24px; }}
    .network-node {{ align-items: center; background: #eef3ef; border: 1px solid #cad8cd; border-radius: 12px; display: flex; flex-direction: column; justify-content: center; min-height: 120px; padding: 16px; text-align: center; }}
    .network-node.inputs {{ align-items: start; text-align: left; }}
    .network-node.inputs span {{ color: var(--muted); font-family: system-ui, sans-serif; font-size: .85rem; }}
    .network-node strong {{ color: var(--accent); font-family: system-ui, sans-serif; font-size: 1.25rem; }}
    .network-node small {{ color: var(--muted); }}
    .network-node.output {{ background: var(--accent-soft); border-color: #d9b8b0; }}
    .network-arrow {{ align-self: center; color: var(--accent); font-size: 1.6rem; }}
    .network-details, .advanced-panel {{ border-top: 1px solid var(--line); padding: 10px 0; }}
    .network-details summary, .advanced-panel summary {{ color: var(--accent); cursor: pointer; font-family: system-ui, sans-serif; font-weight: 700; }}
    .network-details li {{ margin: 8px 0; }}
    .network-details li span {{ display: block; }}
    .network-details code {{ overflow-wrap: anywhere; }}
    .tensor-kind {{ font-weight: 700; }}
    .indices {{ color: var(--muted); font-size: .86rem; }}
    .network-controls {{ display: flex; flex-wrap: wrap; gap: 14px; margin: 22px 0 16px; }}
    .network-controls label {{ color: var(--muted); display: grid; flex: 1 1 280px; font-family: system-ui, sans-serif; font-size: .82rem; gap: 5px; }}
    .network-controls select {{ background: white; border: 1px solid var(--line); border-radius: 8px; color: var(--ink); font: inherit; padding: 10px 12px; width: 100%; }}
    .network-legend {{ display: flex; flex-wrap: wrap; gap: 10px 18px; margin: 10px 0 18px; }}
    .network-legend > span {{ align-items: center; color: var(--muted); display: inline-flex; font-family: system-ui, sans-serif; font-size: .82rem; gap: 6px; }}
    .legend-node {{ border: 1px solid; border-radius: 50%; display: inline-block; height: 19px; width: 30px; }}
    .legend-node.source {{ background: #dce9ef; border-color: #64869a; }}
    .legend-node.sink {{ background: #f0dfda; border-color: #a66b60; }}
    .legend-line {{ border-top: 3px solid; display: inline-block; width: 28px; }}
    .legend-line.quark {{ border-color: #315b73; }}
    .legend-line.open-spin {{ border-color: #8b4875; border-top-style: dashed; }}
    .tensor-canvas-shell {{ background: #fff; border: 1px solid var(--line); border-radius: 12px; overflow-x: auto; }}
    #tensor-network-canvas {{ display: block; height: auto; min-width: 780px; width: 100%; }}
    .quark-line {{ opacity: .9; stroke: #315b73; stroke-linecap: round; stroke-linejoin: round; stroke-width: 3; }}
    .quark-line.flavor-d {{ stroke: #a14d36; }}
    .quark-line.flavor-s {{ stroke: #557a55; }}
    .quark-line.flavor-c {{ stroke: #8c651f; }}
    .quark-line.flavor-b {{ stroke: #493d70; }}
    .quark-label {{ fill: #24485c; font-family: system-ui, sans-serif; font-size: 12px; font-weight: 700; paint-order: stroke; stroke: white; stroke-width: 5px; }}
    .quark-label.flavor-d {{ fill: #8d3f2d; }}
    .quark-label.flavor-s {{ fill: #456945; }}
    .quark-label.flavor-c {{ fill: #765317; }}
    .quark-label.flavor-b {{ fill: #493d70; }}
    .operator-node ellipse {{ stroke-width: 1.7; }}
    .source-operator ellipse {{ fill: #dce9ef; stroke: #64869a; }}
    .sink-operator ellipse {{ fill: #f0dfda; stroke: #a66b60; }}
    .insertion-operator ellipse {{ fill: #f0ead8; stroke: #a9894e; }}
    .operator-role {{ fill: var(--muted); font-family: system-ui, sans-serif; font-size: 12px; font-weight: 700; }}
    .operator-label {{ fill: var(--ink); font-family: Georgia, "Times New Roman", serif; font-size: 18px; font-weight: 700; }}
    .operator-feature {{ fill: #6b6258; font-family: system-ui, sans-serif; font-size: 11px; font-style: italic; }}
    .open-spin-line {{ stroke: #8b4875; stroke-dasharray: 6 4; stroke-width: 2.5; }}
    .open-spin-label {{ fill: #71365e; font-family: system-ui, sans-serif; font-size: 10px; font-weight: 700; }}
    .component-title {{ fill: var(--muted); font-family: system-ui, sans-serif; font-size: 13px; font-weight: 700; }}
    .empty-network {{ fill: var(--muted); font-family: system-ui, sans-serif; font-size: 14px; }}
    #network-caption {{ background: #f2eee5; border-radius: 8px; font-family: system-ui, sans-serif; margin: 12px 0 0; padding: 10px 12px; }}
    .network-audit > summary {{ color: var(--accent); cursor: pointer; font-family: system-ui, sans-serif; font-size: 1rem; font-weight: 750; }}
    .network-audit-body {{ margin-top: 20px; }}
    .network-audit-body h2 {{ margin-top: 26px; }}
    .runtime-overview {{ margin-top: 0; }}
    .runtime-insight {{ background: var(--formula); border-left: 4px solid var(--green); border-radius: 8px; color: var(--ink); font-family: system-ui, sans-serif; font-weight: 650; margin: 20px 0 0; padding: 12px 14px; }}
    .stacked-time {{ background: #e5dfd3; border-radius: 999px; display: flex; height: 24px; margin: 26px 0; overflow: hidden; width: 100%; }}
    .segment {{ min-width: 1px; }}
    .stage-0 {{ background: #315b73; }} .stage-1 {{ background: #557a55; }} .stage-2 {{ background: #7b2d26; }} .stage-3 {{ background: #a47128; }} .stage-4 {{ background: #6c4f8a; }} .stage-5 {{ background: #3f7f7a; }} .stage-6 {{ background: #8b8276; }}
    .timing-row {{ align-items: center; border-bottom: 1px solid var(--line); display: grid; gap: 20px; grid-template-columns: 1fr auto; padding: 12px 0; }}
    .timing-label {{ align-items: start; display: grid; gap: 10px; grid-template-columns: auto 1fr; }}
    .timing-label small, .timing-value span {{ color: var(--muted); display: block; font-family: system-ui, sans-serif; font-size: .82rem; }}
    .stage-key {{ border-radius: 999px; display: inline-block; height: 12px; margin-top: .42rem; width: 12px; }}
    .timing-value {{ min-width: 110px; text-align: right; }}
    .one-time-cost {{ align-items: center; background: #f2eee5; border-radius: 8px; display: grid; gap: 4px 14px; grid-template-columns: 1fr auto; margin-top: 18px; padding: 14px; }}
    .one-time-cost small {{ color: var(--muted); grid-column: 1 / -1; }}
    .record-note {{ border-left: 3px solid var(--gold); margin-top: 18px; padding-left: 12px; }}
    .advanced-panel {{ margin-top: 18px; }}
    footer {{ color: var(--muted); font-family: system-ui, sans-serif; font-size: .86rem; margin-top: 34px; }}
    @media (max-width: 820px) {{ .network-flow {{ grid-template-columns: 1fr; }} .network-arrow {{ transform: rotate(90deg); }} }}
    @media (max-width: 620px) {{ main {{ width: min(100% - 20px, 1120px); padding-top: 24px; }} .plain-list > div {{ grid-template-columns: 1fr; gap: 2px; }} .channel-heading {{ display: block; }} .back {{ display: inline-block; margin-top: 8px; }} .timing-row {{ grid-template-columns: 1fr; }} .timing-value {{ text-align: left; }} .view-tabs {{ position: static; }} }}
  </style>
</head>
<body>
<main id="top">
  <header>
    <p class="eyebrow">ConCorD calculation report</p>
    <h1>{title}</h1>
    {description}
    <p class="intro">{html.escape(report_intro)}</p>
  </header>
  <nav class="view-tabs" aria-label="Report views">
    <a class="active" href="#physics" data-view-target="physics">Physics</a>
    <a href="#network" data-view-target="network">Quark flow</a>
    <a href="#runtime" data-view-target="runtime">Runtime</a>
    <a href="#numerical" data-view-target="numerical">Numerical record</a>
  </nav>
  <section class="report-view active" id="physics" data-report-view="physics">
    <section class="overview">{overview_cards}</section>
    <section class="panel">
      <h2>Physics setup</h2>
      <dl class="plain-list">{_definition_list(setup)}</dl>
    </section>
    <section class="panel">
      <h2>Notation</h2>
      <p class="note">Repeated color and spin indices are summed. Bold symbols denote spatial lattice positions. Momentum projection uses ConCorD's convention <strong>exp(+2π i n·x/L)</strong>. Open indices remain on the operator or correlator.</p>
    </section>
    <nav class="panel" aria-label="Correlator channels">
      <h2>Correlator channels</h2>
      <div class="channel-index">{channel_links}</div>
      <label for="channel-search" class="note">Filter channels</label>
      <input class="channel-search" id="channel-search" type="search" placeholder="Name, momentum, or channel label">
    </nav>
    {''.join(channel_sections)}
  </section>
  <section class="report-view" id="network" data-report-view="network">
    {network_html}
  </section>
  <section class="report-view" id="runtime" data-report-view="runtime">
    {runtime_html}
  </section>
  <section class="report-view" id="numerical" data-report-view="numerical">
    {numerical_html}
  </section>
  <footer>Generated from the validated ConCorD job, compiled plan, and optional run record. Detailed runtime traces remain separate from this shareable report.</footer>
</main>
<script>
  document.documentElement.classList.add("js");
  const views = [...document.querySelectorAll("[data-report-view]")];
  const tabs = [...document.querySelectorAll("[data-view-target]")];
  function showView(name, updateHash = false) {{
    const selected = views.some(view => view.dataset.reportView === name) ? name : "physics";
    views.forEach(view => view.classList.toggle("active", view.dataset.reportView === selected));
    tabs.forEach(tab => tab.classList.toggle("active", tab.dataset.viewTarget === selected));
    if (updateHash) history.replaceState(null, "", `#${{selected}}`);
  }}
  tabs.forEach(tab => tab.addEventListener("click", event => {{
    event.preventDefault();
    showView(tab.dataset.viewTarget, true);
  }}));
  const requested = location.hash.slice(1);
  if (requested.startsWith("channel-")) {{
    showView("physics");
    document.getElementById(requested)?.scrollIntoView();
  }} else {{
    showView(requested || "physics");
  }}
  document.getElementById("channel-search")?.addEventListener("input", event => {{
    const query = event.target.value.trim().toLowerCase();
    document.querySelectorAll("[data-channel-search]").forEach(channel => {{
      channel.hidden = query.length > 0 && !channel.dataset.channelSearch.includes(query);
    }});
  }});
  async function copyText(value) {{
    if (navigator.clipboard?.writeText) {{
      await navigator.clipboard.writeText(value);
      return;
    }}
    const temporary = document.createElement("textarea");
    temporary.value = value;
    temporary.style.position = "fixed";
    temporary.style.opacity = "0";
    document.body.appendChild(temporary);
    temporary.select();
    document.execCommand("copy");
    temporary.remove();
  }}
  document.querySelectorAll(".copy-latex").forEach(button => {{
    button.addEventListener("click", async () => {{
      const source = button.parentElement.querySelector("code")?.textContent || "";
      try {{
        await copyText(source);
        const original = button.textContent;
        button.textContent = "Copied";
        setTimeout(() => {{ button.textContent = original; }}, 1200);
      }} catch (error) {{
        button.textContent = "Copy failed";
      }}
    }});
  }});
</script>
</body>
</html>
"""


def write_job_visualization(
    spec: JobSpec,
    collection: OperatorCollection,
    path: str | Path,
    *,
    plan: ConcordPlan | None = None,
    run_record: JobRunRecord | None = None,
    overwrite: bool = False,
) -> Path:
    """Atomically write a standalone HTML calculation report."""

    target = Path(path).expanduser().resolve(strict=False)
    if target.suffix.lower() not in {".html", ".htm"}:
        raise PhysicsVisualizationError(
            "physics visualization output must use .html or .htm"
        )
    if target.exists() and not overwrite:
        raise FileExistsError(f"physics visualization already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            render_job_visualization(
                spec,
                collection,
                plan=plan,
                run_record=run_record,
            ),
            encoding="utf-8",
        )
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


__all__ = [
    "Formula",
    "PhysicsVisualizationError",
    "latex_correlator",
    "latex_expression",
    "latex_operator",
    "render_job_visualization",
    "write_job_visualization",
]
