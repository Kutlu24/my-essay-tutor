"""Exam catalog: which exam frameworks each language can be graded against,
their levels and writing tasks, and the deterministic exam-native scoring
that runs *after* the LLM has scored the four/five schema criteria.

Sources (official rating documents, read 2026-10-10):
- DELF A1-B2 / DALF C1 "Grille d'evaluation de la production ecrite"
  (France Education international, france-education-international.fr/document/
  grille-pe-{a1,a2,b1,b2,c1}). Five criteria (realisation de la tache,
  coherence et cohesion, adequation sociolinguistique, lexique,
  morphosyntaxe), four performance levels each (insufficient / below the
  target level / at the level / level+), fixed point values per band, and
  anomaly rules (off-topic caps, blank paper, under 50% of the expected
  words = 0). The DALF C1 synthese grid has a different criteria set
  (objectivity instead of sociolinguistic adequacy, plus a length criterion).
  The DALF C2 grid could not be retrieved and is not offered.
- Goethe-Zertifikat A2 / B1 / B2 / C1 "Bewertungskriterien Schreiben" and
  Bewertungsbogen (goethe.de Modellsatz PDFs). B1-C1: Aufgabenerfuellung,
  Kohaerenz, Wortschatz, Strukturen, bands A-E, per-task point weights; A2:
  Aufgabenerfuellung + Sprache. An E on Aufgabenerfuellung zeroes the task.
  Goethe A1 (Start Deutsch 1) is not offered - its rating sheet was not found.

The LLM scores each criterion 0-12 (the project's universal scale); this
module quantizes that into the exam's official bands and point values. The
quantization thresholds are this project's own choice, not the exam
bodies', so exam-native points are an estimate, not an examiner's result.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import CriteriaScores, ExamCriterionResult, ExamScore

# CriteriaScores field -> shown for the LLM in the JSON schema.
CRITERION_KEYS = ("vocabulary", "coherence", "grammar", "content_relevance", "sociolinguistic", "objectivity")


@dataclass(frozen=True)
class CriterionSpec:
    label: str  # the exam body's own name for the criterion
    keys: tuple[str, ...]  # CriteriaScores fields this criterion is read from (averaged)
    points: tuple[float, ...]  # points per band, best band first


@dataclass(frozen=True)
class Task:
    id: str
    label: str
    expected_words: str  # human-readable, e.g. "~250"
    zero_below: int  # strictly fewer words than this -> the whole task scores 0
    criteria: tuple[CriterionSpec, ...]
    brief: str  # what the candidate was asked to write (for the prompt)
    rubric: str  # official criteria/descriptors summary (for the prompt)
    note: str = ""  # key into _MSG: shown with the result (pass line, what's not assessed)


@dataclass(frozen=True)
class Exam:
    id: str
    label: str
    languages: tuple[str, ...]
    scheme: str  # "goethe" (bands A-E) | "delf" (4 performance levels) | "generic"
    levels: dict[str, tuple[Task, ...]]


# ---------------------------------------------------------------- DELF/DALF

_DELF_COMMON = """Official FEI grid (France Education international), criteria graded on four performance levels: insufficient / below the target level / target level / target level+ (e.g. B2 / B2+):
- Realisation de la tache -> content_relevance: task and text type fulfilled, instructions followed, ideas developed.
- Coherence et cohesion -> coherence: organisation, paragraphing, connectors, logical progression.
- Adequation sociolinguistique -> sociolinguistic: register and conventions suited to the situation and addressee (tu/vous, formulas, salutations).
- Lexique -> vocabulary: range and precision of vocabulary for the level.
- Morphosyntaxe -> grammar: control of morphology, syntax, agreement, spelling at the level.
Anomalies (apply them): off-topic content (thematic and/or discursive) caps the task-achievement, coherence and lexique criteria; a production under 50% of the expected length scores 0."""

# Task.note holds a key into _MSG (localized when the result is built).
_DELF_NOTE = "delf"


def _delf_criteria(pts: tuple[float, ...]) -> tuple[CriterionSpec, ...]:
    return (
        CriterionSpec("Réalisation de la tâche", ("content_relevance",), pts),
        CriterionSpec("Cohérence et cohésion", ("coherence",), pts),
        CriterionSpec("Adéquation sociolinguistique", ("sociolinguistic",), pts),
        CriterionSpec("Lexique", ("vocabulary",), pts),
        CriterionSpec("Morphosyntaxe", ("grammar",), pts),
    )


def _delf_task(pts, expected, zero_below, brief, label="Production écrite") -> Task:
    return Task(
        id="pe", label=label, expected_words=expected, zero_below=zero_below,
        criteria=_delf_criteria(pts), brief=brief, rubric=_DELF_COMMON, note=_DELF_NOTE,
    )


_DALF_SYNTHESE = Task(
    id="synthese", label="Synthèse", expected_words="~200", zero_below=100,
    criteria=(
        CriterionSpec("Respect de la règle d'objectivité", ("objectivity",), (1.5, 1.0, 0.5, 0.0)),
        CriterionSpec("Réalisation de la tâche : synthèse", ("content_relevance",), (2.5, 1.5, 1.0, 0.0)),
        CriterionSpec("Cohérence et cohésion", ("coherence",), (2.5, 1.5, 1.0, 0.0)),
        CriterionSpec("Lexique", ("vocabulary",), (3.0, 2.0, 1.0, 0.0)),
        CriterionSpec("Morphosyntaxe", ("grammar",), (2.5, 1.5, 1.0, 0.0)),
    ),
    brief=(
        "A synthese (summary of several source documents, ~1000 words in total) of about 200 words: "
        "objective, restructured around the documents' common theme, no personal opinion, no copying passages."
    ),
    rubric="""Official DALF C1 grid, production ecrite - SYNTHESE (criteria graded insufficient / below C1 / C1 / C1+):
- Respect de la regle d'objectivite -> objectivity: the synthese reports the documents' content without the writer's own opinion, judgement or additions.
- Realisation de la tache : synthese -> content_relevance: the main ideas of ALL documents identified and reformulated into a restructured whole (not a document-by-document paraphrase), faithful to the sources.
- Coherence et cohesion -> coherence: clear organisation, varied connectors, logical progression, cohesive devices.
- Lexique -> vocabulary: wide, precise, reformulated vocabulary; avoid copying the documents' wording.
- Morphosyntaxe -> grammar: complex structures controlled; only occasional slips.
Anomalies: off-topic caps task/lexique/coherence; copying whole passages of the documents is penalised; 300 words or more means the task cannot be rated C1 or C1+; under 100 words scores 0.""",
    note="dalf_synthese",
)

_DALF_ESSAI = Task(
    id="essai", label="Essai argumenté", expected_words="≥250", zero_below=125,
    criteria=_delf_criteria((2.5, 1.5, 0.5, 0.0)),
    brief=(
        "An essai argumente (argumentative essay, at least 250 words) taking a position on the "
        "topic raised by the documents, with a structured argumentation and a conclusion."
    ),
    rubric=_DELF_COMMON.replace("B2 / B2+", "C1 / C1+")
    + "\nC1 expects a nuanced, well-structured argument with introduction, developed arguments, concession/counter-argument and conclusion; "
    "an academic/formal register held throughout.",
    note=_DELF_NOTE,
)

DELF = Exam(
    id="delf", label="DELF (A1–B2)", languages=("fr",), scheme="delf",
    levels={
        "A1": (_delf_task((3.0, 2.0, 0.5, 0.0), "~40", 20,
                          "Exercice 2: a short, simple everyday message (invitation, note, postcard) of about 40 words.",
                          "Exercice 2"),),
        "A2": (_delf_task((2.5, 1.5, 0.5, 0.0), "~60", 30,
                          "A short everyday message or letter of about 60 words (invitation, thanks, apology, request).",
                          "Exercice 1 / 2"),),
        "B1": (_delf_task((5.0, 3.0, 1.0, 0.0), "~160", 80,
                          "A personal letter/message or short opinion text of about 160 words on a familiar topic."),),
        "B2": (_delf_task((5.0, 3.0, 1.0, 0.0), "~250", 125,
                          "An essai argumente / contribution to a debate (letter to an editor, forum post) of about 250 words, "
                          "taking a clear position with arguments and examples."),),
    },
)

DALF = Exam(
    id="dalf", label="DALF (C1)", languages=("fr",), scheme="delf",
    levels={"C1": (_DALF_ESSAI, _DALF_SYNTHESE)},
)

# ------------------------------------------------------------------ Goethe

_GOETHE_RUBRIC = """Official Goethe-Zertifikat "Bewertungskriterien Schreiben" - each criterion is rated A (best) to E (0 points):
- Aufgabenerfuellung -> content_relevance: Inhalt, Umfang, Realisierung der Sprachfunktionen ({functions}) - A: all language functions content- and length-appropriate; B: one fewer; C: about half; D: one only partly; E: under 50% of the required words or topic missed. Plus Register/soziokulturelle Angemessenheit: situations- und partneradaequat (A) down to no longer appropriate.
- Kohaerenz -> coherence: Textaufbau (Einleitung, Schluss, Logik) durchgaengig und effektiv (A) ... kaum erkennbar; Verknuepfung von Saetzen und Satzteilen angemessen/flexibel ... kaum angemessen.
- Wortschatz -> vocabulary: Spektrum (differenziert/breit -> kaum vorhanden) and Beherrschung (vereinzelte Fehlgriffe beeintraechtigen das Verstaendnis nicht (A) ... erheblich).
- Strukturen -> grammar: Spektrum and Beherrschung (Morphologie, Syntax, Orthografie), same scale.
An E on Aufgabenerfuellung makes the whole task worth 0 points."""

_GOETHE_A2_RUBRIC = """Official Goethe-Zertifikat A2 "Bewertungskriterien Schreiben" - two criteria, each rated A (best) to E (0 points):
- Aufgabenerfuellung -> content_relevance: all 3 Sprachfunktionen inhaltlich und umfaenglich angemessen (A) / 2 (B) / 1 + 1 teilweise (C) / 1 (D) / Textumfang under 50% or Thema verfehlt (E); plus Register: situations- und partneradaequat.
- Sprache -> coherence + vocabulary + grammar together: Spektrum (angemessen und differenziert -> kaum angemessen) and Beherrschung (vereinzelte Fehlgriffe beeintraechtigen das Verstaendnis nicht -> erheblich).
An E on Aufgabenerfuellung makes the whole task worth 0 points."""

_GOETHE_NOTE = "goethe"

_G_BANDS_FULL = (1.0, 0.75, 0.5, 0.25, 0.0)


def _g_task(tid: str, label: str, expected: str, zero_below: int, brief: str, weights: tuple[int, int, int, int],
            functions: str) -> Task:
    w_e, w_k, w_w, w_s = weights

    def pts(w: int) -> tuple[float, ...]:
        return tuple(w * f for f in _G_BANDS_FULL)

    return Task(
        id=tid, label=label, expected_words=expected, zero_below=zero_below,
        criteria=(
            CriterionSpec("Aufgabenerfüllung", ("content_relevance",), pts(w_e)),
            CriterionSpec("Kohärenz", ("coherence",), pts(w_k)),
            CriterionSpec("Wortschatz", ("vocabulary",), pts(w_w)),
            CriterionSpec("Strukturen", ("grammar",), pts(w_s)),
        ),
        brief=brief, rubric=_GOETHE_RUBRIC.format(functions=functions), note=_GOETHE_NOTE,
    )


def _g_a2_task(tid: str, label: str, expected: str, zero_below: int, brief: str) -> Task:
    a2 = (5.0, 3.5, 2.0, 0.5, 0.0)  # official A2 sheet: not linear
    return Task(
        id=tid, label=label, expected_words=expected, zero_below=zero_below,
        criteria=(
            CriterionSpec("Aufgabenerfüllung", ("content_relevance",), a2),
            CriterionSpec("Sprache", ("coherence", "vocabulary", "grammar"), a2),
        ),
        brief=brief, rubric=_GOETHE_A2_RUBRIC, note=_GOETHE_NOTE,
    )


GOETHE = Exam(
    id="goethe", label="Goethe-Zertifikat (A2–C1)", languages=("de",), scheme="goethe",
    levels={
        "A2": (
            _g_a2_task("t1", "Teil 1 – kurze Mitteilung", "20–30", 10,
                       "A short informal message (SMS/note) of 20-30 words covering three given points."),
            _g_a2_task("t2", "Teil 2 – E-Mail", "30–40", 15,
                       "A short e-mail of 30-40 words covering three given points."),
        ),
        "B1": (
            _g_task("t1", "Teil 1 – persönliche E-Mail", "~80", 40,
                    "A personal e-mail of about 80 words covering three given points (describe, give reasons, make a suggestion).",
                    (10, 10, 10, 10), "3 Sprachfunktionen, z. B. jemanden einladen, Vorschlag machen"),
            _g_task("t2", "Teil 2 – Meinungsbeitrag", "~80", 40,
                    "A forum/discussion contribution of about 80 words giving a personal opinion on a topic.",
                    (10, 10, 10, 10), "Meinungsäußerung inhaltlich und umfänglich angemessen"),
            _g_task("t3", "Teil 3 – kurze Mitteilung", "~40", 20,
                    "A short, polite message of about 40 words (apologise, ask for something) to a course leader or similar.",
                    (4, 4, 6, 6), "Mitteilung, Inhalt"),
        ),
        "B2": (
            _g_task("t1", "Teil 1 – Forumsbeitrag", "~150", 75,
                    "A discussion-forum contribution of about 150 words on a topic, stating and justifying an opinion.",
                    (14, 14, 16, 16), "4 Sprachfunktionen, z. B. Meinung äußern, begründen, Beispiele nennen"),
            _g_task("t2", "Teil 2 – formelle E-Mail", "~100", 50,
                    "A semi-formal/formal e-mail of about 100 words (e.g. apologise, express regret, ask for something).",
                    (10, 10, 10, 10), "4 Sprachfunktionen, z. B. sich entschuldigen, Bedauern ausdrücken, um etwas bitten"),
        ),
        "C1": (
            _g_task("t1", "Teil 1 – Diskussionsbeitrag", "~230", 115,
                    "A discussion contribution of about 230 words: argue a position with differentiated reasoning.",
                    (14, 14, 16, 16), "4 Sprachfunktionen, z. B. etwas erklären, Argumente anführen, Vorschlag machen"),
            _g_task("t2", "Teil 2 – formelle E-Mail", "~120", 60,
                    "A formal e-mail of about 120 words, register-appropriate and precise.",
                    (10, 10, 10, 10), "4 Sprachfunktionen"),
        ),
    },
)

# --------------------------------------------------- legacy / generic paths

TELC = Exam(
    id="telc", label="telc Deutsch (A1–C1)", languages=("de",), scheme="generic",
    levels={lvl: () for lvl in ("A1", "A2", "B1", "B2", "C1")},
)
CEFR = Exam(
    id="cefr", label="General CEFR (A1–C1)", languages=("en", "fr"), scheme="generic",
    levels={lvl: () for lvl in ("A1", "A2", "B1", "B2", "C1")},
)

EXAMS: dict[str, Exam] = {e.id: e for e in (TELC, GOETHE, DELF, DALF, CEFR)}


def default_exam(language: str, level: str) -> str:
    if language == "de":
        return "telc"
    if language == "fr":
        return "dalf" if level == "C1" else "delf"
    return "cefr"


def resolve(language: str, exam_id: str | None, level: str, task_id: str | None) -> tuple[Exam, Task | None]:
    """Validate a (language, exam, level, task) selection; raise ValueError if invalid."""
    exam = EXAMS.get(exam_id or default_exam(language, level))
    if exam is None:
        raise ValueError(f"Unknown exam: {exam_id}")
    if language not in exam.languages:
        raise ValueError(f"{exam.id} is not available for language '{language}'")
    if level not in exam.levels:
        raise ValueError(f"{exam.id} does not offer level {level}; choose one of {', '.join(exam.levels)}")
    tasks = exam.levels[level]
    if not tasks:
        return exam, None
    if task_id is None:
        return exam, tasks[0]
    for t in tasks:
        if t.id == task_id:
            return exam, t
    raise ValueError(f"Unknown task '{task_id}' for {exam.id} {level}; choose one of {', '.join(t.id for t in tasks)}")


def catalog() -> dict:
    """JSON-ready listing for the frontend: language -> exams -> levels -> tasks."""
    out: dict[str, list] = {"de": [], "en": [], "fr": []}
    for exam in EXAMS.values():
        for lang in exam.languages:
            out[lang].append({
                "id": exam.id,
                "label": exam.label,
                "levels": [
                    {"level": lvl, "tasks": [
                        {"id": t.id, "label": t.label, "expected_words": t.expected_words} for t in tasks
                    ]}
                    for lvl, tasks in exam.levels.items()
                ],
            })
    return out


# ------------------------------------------------------------------ scoring

_GOETHE_LETTERS = ("A", "B", "C", "D", "E")

_MSG = {
    "en": {
        "delf": "DELF/DALF: a diploma needs 50/100 over all four skills and at least 5/25 in each skill - a single written production cannot decide it.",
        "dalf_synthese": "Maximum shown is 12/12.5: the grid's length-compliance criterion (0.5) is not assessed. DELF/DALF: a diploma needs 50/100 over all four skills and at least 5/25 in each skill.",
        "goethe": "Goethe: the module needs 60/100 over all Schreiben tasks; the points shown are for this one task only.",
        "zero_words": "{n} words is under 50% of the expected length ({exp}): the task scores 0.",
        "off_topic": "Off-topic ({kind}) caps the affected criteria, as in the official grid.",
        "goethe_e": "Aufgabenerfüllung is E: the whole task scores 0 points (official rule).",
    },
    "de": {
        "delf": "DELF/DALF: Für das Diplom braucht es 50/100 über alle vier Fertigkeiten und mindestens 5/25 pro Fertigkeit – ein einzelner Schreibtext kann das nicht entscheiden.",
        "dalf_synthese": "Angezeigtes Maximum: 12/12,5 – das Kriterium zur Längenvorgabe (0,5) wird nicht bewertet. DELF/DALF: Für das Diplom braucht es 50/100 über alle vier Fertigkeiten und mindestens 5/25 pro Fertigkeit.",
        "goethe": "Goethe: Für das Modul braucht es 60/100 über alle Schreiben-Aufgaben; die angezeigten Punkte gelten nur für diese eine Aufgabe.",
        "zero_words": "{n} Wörter sind weniger als 50 % der erwarteten Länge ({exp}): Die Aufgabe erhält 0 Punkte.",
        "off_topic": "Themaverfehlung ({kind}) begrenzt die betroffenen Kriterien, wie im offiziellen Raster.",
        "goethe_e": "Aufgabenerfüllung ist E: Die ganze Aufgabe erhält 0 Punkte (offizielle Regel).",
    },
    "fr": {
        "delf": "DELF/DALF : le diplôme exige 50/100 sur les quatre compétences et au moins 5/25 dans chacune – une seule production écrite ne peut pas le décider.",
        "dalf_synthese": "Maximum affiché : 12/12,5 – le critère de respect de la consigne de longueur (0,5) n'est pas évalué. DELF/DALF : le diplôme exige 50/100 sur les quatre compétences et au moins 5/25 dans chacune.",
        "goethe": "Goethe : le module exige 60/100 sur toutes les tâches de Schreiben ; les points affichés ne concernent que cette tâche.",
        "zero_words": "{n} mots, c'est moins de 50 % de la longueur attendue ({exp}) : la tâche vaut 0.",
        "off_topic": "Un hors-sujet ({kind}) plafonne les critères concernés, comme dans la grille officielle.",
        "goethe_e": "Aufgabenerfüllung est E : toute la tâche vaut 0 point (règle officielle).",
    },
}


def _band_index(score: int, n_bands: int) -> int:
    """0 = best band. Thresholds on the 0-12 LLM scale are this project's choice."""
    thresholds = (11, 8, 5, 2) if n_bands == 5 else (11, 7, 2)
    for i, t in enumerate(thresholds):
        if score >= t:
            return i
    return n_bands - 1


def _delf_labels(level: str) -> tuple[str, ...]:
    return (f"{level}+", level, f"< {level}", "insuffisant")


# off-topic cap: criterion key -> worst-allowed... (min band index, 0 = best)
_OFF_TOPIC_MIN_BAND = {
    "thematic": {"content_relevance": 1, "vocabulary": 1},
    "discursive": {"content_relevance": 2, "coherence": 2},
    "complete": {"content_relevance": 3, "coherence": 3, "sociolinguistic": 3, "vocabulary": 2, "grammar": 2},
}


def score_task(exam: Exam, level: str, task: Task, criteria: CriteriaScores, word_count: int,
               off_topic: str, lang: str = "en") -> ExamScore:
    n_bands = len(task.criteria[0].points)
    labels = _GOETHE_LETTERS if exam.scheme == "goethe" else _delf_labels(level)
    max_total = sum(c.points[0] for c in task.criteria)
    notes: list[str] = []
    msg = _MSG.get(lang, _MSG["en"])

    def result(rows: list[ExamCriterionResult], reason: str | None = None) -> ExamScore:
        total = sum(r.points for r in rows)
        return ExamScore(
            exam=exam.label, level=level, task=task.label, criteria=rows,
            total=total, max_total=max_total, percent=round(100 * total / max_total) if max_total else 0,
            adjustments=([reason] if reason else []) + notes, note=msg.get(task.note, ""),
        )

    if word_count < task.zero_below:
        rows = [ExamCriterionResult(label=c.label, band=labels[-1], points=0.0, max_points=c.points[0])
                for c in task.criteria]
        return result(rows, msg["zero_words"].format(n=word_count, exp=task.expected_words))

    caps = _OFF_TOPIC_MIN_BAND.get(off_topic, {}) if exam.scheme == "delf" else {}
    if exam.scheme == "delf" and off_topic != "none":
        notes.append(msg["off_topic"].format(kind=off_topic))
    rows = []
    for spec in task.criteria:
        raw = round(sum(getattr(criteria, k).score for k in spec.keys) / len(spec.keys))
        idx = _band_index(raw, n_bands)
        for k in spec.keys:
            idx = max(idx, caps.get(k, 0))
        rows.append(ExamCriterionResult(label=spec.label, band=labels[idx], points=spec.points[idx], max_points=spec.points[0]))

    if exam.scheme == "goethe":
        if rows[0].band == "E" or off_topic == "complete":
            rows = [r.model_copy(update={"points": 0.0, "band": "E"}) for r in rows]
            return result(rows, msg["goethe_e"])
    return result(rows)
