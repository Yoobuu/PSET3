from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from textwrap import indent


EXCLUDED_DIRS = {".git", "node_modules", ".venv", "__pycache__", ".ipynb_checkpoints"}
EXPECTED_NOTEBOOKS = [
    "01_ingesta_parquet_raw.ipynb",
    "02_enriquecimiento_y_unificacion.ipynb",
    "03_construccion_obt.ipynb",
    "04_validaciones_y_exploracion.ipynb",
    "05_data_analysis.ipynb",
]
ENV_KEYS_PRIMARY = [
    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_HOST",
    "SNOWFLAKE_USER",
    "SNOWFLAKE_PASSWORD",
    "SNOWFLAKE_DATABASE",
    "SNOWFLAKE_SCHEMA_RAW",
    "SNOWFLAKE_SCHEMA_ANALYTICS",
    "SNOWFLAKE_WAREHOUSE",
    "SNOWFLAKE_ROLE",
]
ENV_KEYS_DATA = [
    "DATA_BASE_URL",
    "DATA_ROOT",
    "YEARS",
    "MONTHS",
    "SERVICES",
    "TAXI_ZONE_LOOKUP_PATH",
    "CHUNK",
    "RUN_ID",
    "ENABLE_RANGE_CHECKS",
]
REQUIRED_FOR_SNOWFLAKE = [
    "SNOWFLAKE_USER",
    "SNOWFLAKE_PASSWORD",
    "SNOWFLAKE_DATABASE",
    "SNOWFLAKE_SCHEMA_RAW",
    "SNOWFLAKE_SCHEMA_ANALYTICS",
    "SNOWFLAKE_WAREHOUSE",
]
REQUIRED_FOR_SNOWFLAKE_ACCOUNT = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_HOST"]
DEPENDENCY_KEYWORDS = [
    "pyspark",
    "snowflake-connector-python",
    "snowflake-snowpark-python",
    "pyarrow",
    "net.snowflake:spark-snowflake",
    "snowflake-jdbc",
]
CODE_SEARCH_KEYWORDS = [
    "Snowflake",
    "snowflake",
    "spark.write",
    "spark.read.parquet",
    "sfOptions",
    "COPY INTO",
    "PUT ",
    "jdbc",
]
PARQUET_PATTERN = re.compile(r"(yellow|green)_tripdata_(\d{4})-(\d{2})", re.IGNORECASE)
YEARS_EXPECTED = list(range(2015, 2026))
MONTHS_EXPECTED = [f"{i:02d}" for i in range(1, 13)]


def generate_tree(root: Path, max_depth: int = 3) -> str:
    lines: list[str] = ["."]

    def walk(path: Path, depth: int, prefix: str) -> None:
        if depth >= max_depth:
            return
        try:
            entries = sorted(
                (p for p in path.iterdir() if p.name not in EXCLUDED_DIRS),
                key=lambda x: (not x.is_dir(), x.name.lower()),
            )
        except FileNotFoundError:
            return
        for index, entry in enumerate(entries):
            connector = "`-- " if index == len(entries) - 1 else "|-- "
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{prefix}{connector}{entry.name}{suffix}")
            if entry.is_dir():
                extension = "    " if index == len(entries) - 1 else "|   "
                walk(entry, depth + 1, prefix + extension)

    walk(root, 0, "")
    return "\n".join(lines)


def collect_notebooks(root: Path) -> list[dict[str, object]]:
    found = {}
    search_roots = [root, root / "notebooks", root / "work"]
    for sroot in search_roots:
        if not sroot.exists():
            continue
        for nb_path in sroot.glob("*.ipynb"):
            found[nb_path.resolve()] = nb_path

    notebooks = []
    for nb_path in sorted(found.values(), key=lambda p: p.name):
        try:
            size_bytes = nb_path.stat().st_size
        except OSError:
            continue
        size_kb = size_bytes / 1024
        cell_count = 0
        try:
            with nb_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            cell_count = len(data.get("cells", []))
        except Exception:
            cell_count = -1
        notebooks.append(
            {
                "name": nb_path.name,
                "path": nb_path.relative_to(root).as_posix(),
                "size_kb": round(size_kb, 2),
                "cell_count": cell_count,
                "is_empty": size_bytes < 10 * 1024 or cell_count == 0,
            }
        )
    return notebooks


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="latin-1")
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if "#" in value:
            value = value.split("#", 1)[0].strip()
        values[key] = value
    return values


def parse_compose(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
        services = data.get("services", {}) if isinstance(data, dict) else {}
    except Exception:
        services = {}
        current_service = None
        current_section = None
        for raw_line in text.splitlines():
            if not raw_line.strip():
                continue
            if raw_line.lstrip().startswith("#"):
                continue
            indent_level = len(raw_line) - len(raw_line.lstrip(" "))
            cleaned = raw_line.strip()
            if indent_level == 0 and cleaned.startswith("services"):
                continue
            if indent_level == 2 and cleaned.endswith(":"):
                current_service = cleaned.rstrip(":")
                services[current_service] = {}
                current_section = None
                continue
            if current_service is None:
                continue
            if indent_level == 4 and cleaned.endswith(":"):
                current_section = cleaned.rstrip(":")
                services[current_service].setdefault(current_section, [])
                continue
            if indent_level == 4 and ":" in cleaned and not cleaned.endswith(":"):
                key, val = cleaned.split(":", 1)
                services[current_service][key.strip()] = val.strip().strip('"')
                current_section = None
                continue
            if indent_level >= 6 and cleaned.startswith("-"):
                val = cleaned.lstrip("-").strip().strip('"')
                if current_section is not None:
                    entry = services[current_service].setdefault(current_section, [])
                    entry.append(val)
    normalized = {}
    for sname, sdata in services.items():
        info = {
            "image": None,
            "ports": [],
            "volumes": [],
            "environment": [],
            "env_file": [],
        }
        if isinstance(sdata, dict):
            image = sdata.get("image")
            if isinstance(image, str):
                info["image"] = image
            for key in ("ports", "volumes", "environment", "env_file"):
                val = sdata.get(key)
                if isinstance(val, list):
                    info[key] = [str(v) for v in val]
            env_section = sdata.get("environment")
            if isinstance(env_section, dict):
                info["environment"] = [f"{k}={v}" for k, v in env_section.items()]
        normalized[sname] = info
    return normalized


def find_dependency_files(root: Path) -> list[Path]:
    patterns = [
        "requirements.txt",
        "requirements-dev.txt",
        "requirements/*.txt",
        "pyproject.toml",
        "Pipfile",
        "poetry.lock",
        "environment.yml",
        "environment.yaml",
        "conda.yml",
    ]
    results: set[Path] = set()
    for pattern in patterns:
        if "/" in pattern:
            base, glob_pat = pattern.split("/", 1)
            base_path = root / base
            if base_path.is_dir():
                results.update(base_path.glob(glob_pat))
            continue
        target = root / pattern
        if target.exists():
            results.add(target)
    return sorted(results)


def scan_dependencies(files: list[Path]) -> dict[str, set[str]]:
    findings: dict[str, set[str]] = {}
    for path in files:
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="latin-1")
        lowered = content.lower()
        matches = {dep for dep in DEPENDENCY_KEYWORDS if dep.lower() in lowered}
        findings[path.relative_to(path.parents[1] if path.parents[1] else path.parent)] = matches
    return findings


def inventory_parquet(root: Path) -> dict[str, object]:
    counts: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    files: list[str] = []
    for file in root.rglob("*.parquet"):
        rel = file.relative_to(root).as_posix()
        files.append(rel)
        match = PARQUET_PATTERN.search(file.name)
        if not match:
            continue
        service, year, month = match.groups()
        service = service.lower()
        counts[service][year][month] += 1
    missing: dict[str, dict[str, list[str]]] = {}
    for service in ("yellow", "green"):
        missing[service] = {}
        for year in YEARS_EXPECTED:
            str_year = str(year)
            present_months = counts[service].get(str_year, {})
            absent = [m for m in MONTHS_EXPECTED if present_months.get(m, 0) == 0]
            if absent:
                missing[service][str_year] = absent
    return {
        "counts": counts,
        "files": files,
        "missing": missing,
    }


def search_ingestion_code(root: Path) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    text_extensions = {".py", ".sql", ".scala", ".md"}
    for path in root.rglob("*"):
        if path.is_dir():
            if path.name in EXCLUDED_DIRS:
                continue
            continue
        if path.suffix.lower() in text_extensions:
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = path.read_text(encoding="latin-1")
            for lineno, line in enumerate(content.splitlines(), start=1):
                if any(key in line for key in CODE_SEARCH_KEYWORDS):
                    snippet = line.strip()
                    findings.append(
                        {
                            "path": path,
                            "line": lineno,
                            "snippet": snippet,
                        }
                    )
        elif path.suffix.lower() == ".ipynb":
            try:
                with path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                continue
            for cell_index, cell in enumerate(data.get("cells", [])):
                source = "".join(cell.get("source", []))
                if any(key in source for key in CODE_SEARCH_KEYWORDS):
                    findings.append(
                        {
                            "path": path,
                            "line": cell_index + 1,
                            "snippet": source.strip().splitlines()[0] if source else "",
                        }
                    )
    findings.sort(key=lambda x: (str(x["path"]), x["line"]))
    return findings


def snowflake_status(env_values: dict[str, str]) -> dict[str, object]:
    missing = [key for key in REQUIRED_FOR_SNOWFLAKE if not env_values.get(key)]
    account_present = any(env_values.get(key) for key in REQUIRED_FOR_SNOWFLAKE_ACCOUNT)
    if missing or not account_present:
        reason = "Faltan variables requeridas" if missing else "Falta SNOWFLAKE_ACCOUNT o SNOWFLAKE_HOST"
        details = {
            "missing_variables": missing + ([] if account_present else REQUIRED_FOR_SNOWFLAKE_ACCOUNT),
        }
        return {"status": "skipped", "reason": reason, "details": details}
    try:
        import snowflake.connector  # type: ignore
    except Exception:
        return {
            "status": "skipped",
            "reason": "Paquete snowflake-connector-python no disponible",
            "details": {},
        }
    password = env_values.get("SNOWFLAKE_PASSWORD", "")
    if not password:
        return {
            "status": "skipped",
            "reason": "SNOWFLAKE_PASSWORD vacio; se omite conexion",
            "details": {},
        }
    connect_args = {
        "user": env_values.get("SNOWFLAKE_USER"),
        "password": password,
        "warehouse": env_values.get("SNOWFLAKE_WAREHOUSE"),
        "database": env_values.get("SNOWFLAKE_DATABASE"),
        "schema": env_values.get("SNOWFLAKE_SCHEMA_ANALYTICS"),
    }
    if env_values.get("SNOWFLAKE_ACCOUNT"):
        connect_args["account"] = env_values["SNOWFLAKE_ACCOUNT"]
    elif env_values.get("SNOWFLAKE_HOST"):
        connect_args["host"] = env_values["SNOWFLAKE_HOST"]
    results = {"status": "skipped", "reason": "", "details": {}}
    try:
        connection = snowflake.connector.connect(**connect_args)
    except Exception as exc:
        results["status"] = "error"
        results["reason"] = f"No fue posible conectar: {exc}"
        return results
    try:
        cursor = connection.cursor()
        schemas_raw = cursor.execute("SHOW SCHEMAS").fetchall()
        schema_names = {row[1] for row in schemas_raw}
        raw_schema = env_values.get("SNOWFLAKE_SCHEMA_RAW", "")
        analytics_schema = env_values.get("SNOWFLAKE_SCHEMA_ANALYTICS", "")
        details = {
            "schemas_present": {
                "raw": raw_schema in schema_names,
                "analytics": analytics_schema in schema_names,
            },
            "tables": {},
        }
        if analytics_schema:
            cursor.execute(f"SHOW TABLES IN SCHEMA {analytics_schema}")
            tables = cursor.fetchall()
            table_names = {row[1] for row in tables}
            if "obt_trips" in table_names:
                count = cursor.execute(
                    f"SELECT COUNT(*) FROM {analytics_schema}.obt_trips"
                ).fetchone()
                details["tables"]["analytics.obt_trips"] = count[0] if count else None
        if raw_schema:
            cursor.execute(f"SHOW TABLES IN SCHEMA {raw_schema}")
            raw_tables = cursor.fetchall()
            details["tables"][f"{raw_schema}.*"] = len(raw_tables)
        results["status"] = "ok"
        results["details"] = details
    except Exception as exc:
        results["status"] = "error"
        results["reason"] = f"Error durante comprobaciones: {exc}"
    finally:
        try:
            connection.close()
        except Exception:
            pass
    return results


def format_parquet_table(counts: dict[str, dict[str, dict[str, int]]]) -> str:
    rows = ["| Servicio | Anio | Mes | Archivos |", "|---------|------|-----|----------|"]
    for service in sorted(counts):
        for year in sorted(counts[service]):
            months = counts[service][year]
            for month in sorted(months):
                rows.append(
                    f"| {service} | {year} | {month} | {months[month]} |"
                )
    if len(rows) == 2:
        rows.append("| - | - | - | - |")
    return "\n".join(rows)


def format_missing_months(missing: dict[str, dict[str, list[str]]]) -> str:
    lines: list[str] = []
    for service, years in sorted(missing.items()):
        if not years:
            lines.append(f"- {service}: sin meses faltantes en 2015-2025")
            continue
        for year, months in sorted(years.items()):
            joined = ", ".join(months)
            lines.append(f"- {service} {year}: faltan {joined}")
    return "\n".join(lines)


def build_report(root: Path) -> tuple[str, dict[str, object]]:
    tree = generate_tree(root)
    notebooks = collect_notebooks(root)
    compose_file = root / "docker-compose.yml"
    if not compose_file.exists():
        compose_file = root / "compose.yaml"
    compose_data = parse_compose(compose_file) if compose_file.exists() else {}
    env_values = parse_env_file(root / ".env")
    env_example_values = parse_env_file(root / ".env.example")
    dependency_files = find_dependency_files(root)
    dependency_matches = {}
    if dependency_files:
        dependency_matches = {}
        for path in dependency_files:
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = path.read_text(encoding="latin-1")
            lowered = content.lower()
            hits = sorted(
                {dep for dep in DEPENDENCY_KEYWORDS if dep.lower() in lowered}
            )
            dependency_matches[path.relative_to(root).as_posix()] = hits
    parquet_info = inventory_parquet(root)
    ingestion_hits = search_ingestion_code(root)
    snowflake_info = snowflake_status(env_values)

    notebooks_map = {nb["name"]: nb for nb in notebooks}
    notebooks_section_lines = [
        "| Notebook | Existe | Tamano (KB) | Celdas | Estado | Ruta |",
        "|----------|--------|-------------|--------|--------|------|",
    ]
    for expected in EXPECTED_NOTEBOOKS:
        nb = notebooks_map.get(expected)
        if nb:
            status = "OK" if not nb["is_empty"] else "Vacio"
            notebooks_section_lines.append(
                f"| {expected} | si | {nb['size_kb']} | {nb['cell_count']} | {status} | {nb['path']} |"
            )
        else:
            notebooks_section_lines.append(
                f"| {expected} | no | - | - | falta | - |"
            )
    for nb_name, nb in notebooks_map.items():
        if nb_name not in EXPECTED_NOTEBOOKS:
            status = "OK" if not nb["is_empty"] else "Vacio"
            notebooks_section_lines.append(
                f"| {nb_name} | si | {nb['size_kb']} | {nb['cell_count']} | {status} | {nb['path']} |"
            )
    notebooks_section = "\n".join(notebooks_section_lines)

    env_presence_lines = [
        "| Variable | En .env | En .env.example |",
        "|----------|---------|------------------|",
    ]
    all_env_keys = ENV_KEYS_PRIMARY + [k for k in ENV_KEYS_DATA if k not in ENV_KEYS_PRIMARY]
    seen_keys = set()
    for key in all_env_keys:
        seen_keys.add(key)
        env_presence_lines.append(
            f"| {key} | {'si' if key in env_values else 'no'} | {'si' if key in env_example_values else 'no'} |"
        )
    extra_keys = sorted(set(env_values).difference(seen_keys))
    if extra_keys:
        env_presence_lines.append("| Otros (.env) | si | - |")
    env_section = "\n".join(env_presence_lines)

    compose_section = "No se encontro docker-compose.yml ni compose.yaml."
    if compose_data:
        compose_lines = []
        for service, info in compose_data.items():
            compose_lines.append(f"- **{service}**")
            image = info.get("image") or "sin imagen declarada"
            compose_lines.append(f"  - imagen: {image}")
            ports = ", ".join(info.get("ports", [])) or "sin puertos"
            compose_lines.append(f"  - puertos: {ports}")
            volumes = ", ".join(info.get("volumes", [])) or "sin volumenes"
            compose_lines.append(f"  - volumenes: {volumes}")
            env_vars = ", ".join(info.get("environment", [])) or "sin variables"
            compose_lines.append(f"  - environment: {env_vars}")
            env_file = ", ".join(info.get("env_file", [])) or "sin env_file"
            compose_lines.append(f"  - env_file: {env_file}")
        compose_section = "\n".join(compose_lines)

    dependency_section_lines = []
    if dependency_matches:
        for path, hits in sorted(dependency_matches.items()):
            hit_text = ", ".join(hits) if hits else "sin coincidencias clave"
            dependency_section_lines.append(f"- {path}: {hit_text}")
    else:
        dependency_section_lines.append("- No se identificaron archivos de dependencias.")
    jars_dir = root / "jars"
    if jars_dir.exists():
        jar_files = sorted(p.relative_to(root).as_posix() for p in jars_dir.glob("*.jar"))
        if jar_files:
            dependency_section_lines.append("- JARs detectados:")
            dependency_section_lines.extend(f"  - {jar}" for jar in jar_files)
        else:
            dependency_section_lines.append("- Directorio jars presente pero sin archivos .jar.")
    dependency_section = "\n".join(dependency_section_lines)

    parquet_table = format_parquet_table(parquet_info["counts"])
    parquet_missing = format_missing_months(parquet_info["missing"])
    if not parquet_missing:
        parquet_missing = "- No se encontraron archivos Parquet que coincidan con los patrones esperados."

    ingestion_section_lines = []
    if ingestion_hits:
        for hit in ingestion_hits:
            rel_path = hit["path"].relative_to(root).as_posix()
            snippet = hit["snippet"]
            ingestion_section_lines.append(f"- {rel_path}:{hit['line']} -> {snippet[:120]}")
    else:
        ingestion_section_lines.append("- No se detectaron referencias a Snowflake o escrituras JDBC/Spark.")
    ingestion_section = "\n".join(ingestion_section_lines)

    snowflake_section = "- No se realizo verificacion."
    if snowflake_info["status"] == "ok":
        details = snowflake_info["details"]
        snowflake_section = [
            "- Conexion satisfactoria.",
            f"- Esquema raw presente: {details['schemas_present'].get('raw')}",
            f"- Esquema analytics presente: {details['schemas_present'].get('analytics')}",
        ]
        tables = details.get("tables", {})
        for name, value in tables.items():
            snowflake_section.append(f"- {name}: {value}")
        snowflake_section = "\n".join(snowflake_section)
    elif snowflake_info["status"] == "error":
        reason = snowflake_info.get("reason", "Fallo no especificado")
        snowflake_section = f"- Error al intentar validar: {reason}"
    else:
        reason = snowflake_info.get("reason", "Validacion omitida")
        details = snowflake_info.get("details", {})
        missing_vars = details.get("missing_variables")
        if missing_vars:
            formatted_missing = ", ".join(sorted(set(missing_vars)))
            snowflake_section = f"- Validacion omitida: {reason}. Faltan: {formatted_missing}"
        else:
            snowflake_section = f"- Validacion omitida: {reason}"

    ok_points = []
    warning_points = []
    blocker_points = []
    if compose_data:
        ok_points.append("Docker Compose define un servicio pyspark-notebook con volumen compartido.")
        ports = compose_data.get("pyspark-notebook", {}).get("ports", [])
        if any("8888" in port for port in ports) and any("4040" in port for port in ports):
            ok_points.append("Puertos 8888 y 4040 (via 4148) expuestos para Jupyter y Spark UI.")
        else:
            warning_points.append("Faltan puertos esperados 8888/4040 en docker-compose.")
    else:
        warning_points.append("No se encontro definicion de Docker Compose.")
    empty_env_keys = [key for key in ENV_KEYS_PRIMARY if not env_values.get(key)]
    if empty_env_keys:
        warning_points.append("Variables Snowflake incompletas en .env.")
    else:
        ok_points.append(".env contiene variables principales de Snowflake.")
    if not dependency_matches:
        warning_points.append("No hay archivo de dependencias con pyspark o snowflake-connector.")
    if not ingestion_hits:
        blocker_points.append("No se detecta codigo fuente de ingesta hacia Snowflake.")
    if parquet_info["files"]:
        ok_points.append(f"Inventario local con {len(parquet_info['files'])} archivos Parquet.")
    else:
        blocker_points.append("No hay archivos Parquet locales para ingesta.")

    resumen_lines = []
    for point in ok_points:
        resumen_lines.append(f"- [OK] {point}")
    for point in warning_points:
        resumen_lines.append(f"- [WARN] {point}")
    for point in blocker_points:
        resumen_lines.append(f"- [BLOCK] {point}")
    resumen_section = "\n".join(resumen_lines) if resumen_lines else "- Sin hallazgos."

    plan_lines = [
        "- [ ] Verificar Compose y servicio Jupyter+Spark en docker-compose.",
        "- [ ] Completar archivo .env con credenciales y rutas faltantes.",
        "- [ ] Reprocesar ingesta RAW desde Parquet local sin descargas externas.",
        "- [ ] Construir OBT en Snowflake y ejecutar validaciones de calidad.",
        "- [ ] Documentar evidencias y actualizar README con pasos reproducibles.",
    ]
    plan_section = "\n".join(plan_lines)

    anexos_lines = [
        "- Ruta docker-compose: docker-compose.yml",
        "- Directorio notebooks de trabajo: work/",
        "- Parquet locales: carpeta work/datasets/",
        "- Script de auditoria: scripts/_audit_project.py",
    ]
    anexos_section = "\n".join(anexos_lines)

    bloques_lines = []
    if blocker_points:
        bloques_lines = [f"- {text}" for text in blocker_points]
    else:
        bloques_lines = ["- Sin bloqueos criticos detectados."]

    siguientes_lines = []
    prioritary = warning_points + blocker_points
    if prioritary:
        for point in prioritary:
            siguientes_lines.append(f"- {point}")
    else:
        siguientes_lines.append("- Confirmar estado ejecutando pipelines y documentar.")

    report = f"""# Informe de auditoria P3

## 1) Resumen ejecutivo
{resumen_section}

## 2) Mapa del repo
```
{tree}
```

## 3) Notebooks esperadas y estado
{notebooks_section}

## 4) Docker/Compose
{compose_section}

## 5) Variables y configuracion
{env_section}

## 6) Dependencias
{dependency_section}

## 7) Inventario de Parquet locales
### Archivos contabilizados
{parquet_table}

### Meses faltantes (2015-2025)
{parquet_missing}

## 8) Codigo de ingesta
{ingestion_section}

## 9) Estado contra Snowflake (opcional)
{snowflake_section}

## 10) Plan de reanudacion sugerido
{plan_section}

## 11) Anexos
{anexos_section}

### Bloqueos
{os.linesep.join(bloques_lines)}

### Siguientes acciones prioritarias
{os.linesep.join(siguientes_lines)}
"""
    context = {
        "tree": tree,
        "notebooks": notebooks,
        "compose": compose_data,
        "env": {
            "env": list(env_values.keys()),
            "env_example": list(env_example_values.keys()),
        },
        "dependencies": dependency_matches,
        "parquet": parquet_info,
        "ingestion_hits": [
            {
                "path": str(hit["path"].relative_to(root)),
                "line": hit["line"],
                "snippet": hit["snippet"],
            }
            for hit in ingestion_hits
        ],
        "snowflake": snowflake_info,
    }
    return report, context


def main() -> int:
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    report, context = build_report(project_root)
    report_path = project_root / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    summary_lines = [
        f"Reporte generado en: {report_path}",
        f"Total notebooks auditadas: {len(context['notebooks'])}",
        f"Servicios Docker detectados: {', '.join(context['compose'].keys()) if context['compose'] else 'ninguno'}",
        f"Parquet contabilizados: {len(context['parquet']['files'])}",
        "Snowflake: " + context["snowflake"].get("status", "desconocido"),
    ]
    print("\n".join(summary_lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
