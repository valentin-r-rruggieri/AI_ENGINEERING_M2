from __future__ import annotations

import json

from .config import Settings


SECTIONS = [
    ("Acceso y contraseñas", "Para restablecer una contraseña, la persona debe elegir 'Olvidé mi contraseña' en el inicio de sesión. PeopleFlow envía un enlace de un solo uso al correo laboral registrado. El enlace vence a los treinta minutos y no puede reutilizarse. Si no llega el correo, soporte debe verificar primero que la cuenta exista, que el correo sea correcto y que no esté bloqueada. Las cuentas se bloquean temporalmente después de cinco intentos fallidos y un administrador puede desbloquearlas desde Seguridad."),
    ("Usuarios y roles", "Los administradores crean usuarios desde Configuración, Usuarios, Nuevo usuario. Cada usuario debe tener nombre, correo laboral, país y rol. El rol Administrador administra configuración, usuarios y auditoría. El rol Líder consulta equipos, aprueba licencias y ve reportes de su área. El rol Colaborador consulta su perfil, solicita vacaciones y descarga recibos. Los permisos no se asignan manualmente por pantalla: se heredan desde el rol para evitar configuraciones inconsistentes."),
    ("Onboarding", "El onboarding comienza cuando RR.HH. importa o crea un nuevo empleado. La persona recibe una lista de tareas, documentos requeridos y un contacto responsable. Los documentos sensibles solo pueden cargarse desde una conexión autenticada. El líder debe completar la primera reunión durante los primeros cinco días. RR.HH. puede seguir el avance desde el tablero de onboarding y reenviar recordatorios a quien tenga tareas pendientes."),
    ("Vacaciones", "Las solicitudes de vacaciones se realizan desde Mi tiempo, Nueva solicitud. El sistema calcula el saldo disponible según país, antigüedad y política vigente. La solicitud llega al líder directo y queda aprobada solo después de su confirmación. Si el líder está ausente, el aprobador delegado recibe la tarea. Las solicitudes ya aprobadas pueden cancelarse hasta cuarenta y ocho horas antes de la fecha de inicio, salvo que una política local indique otra condición."),
    ("Licencias", "Las licencias médicas y familiares requieren adjuntar la documentación indicada por la política local. El empleado carga la solicitud y RR.HH. valida los archivos antes de que el saldo se actualice. El líder puede ver el estado, pero no acceder a documentos médicos. Si falta un comprobante, PeopleFlow marca la solicitud como pendiente y envía una notificación. Las licencias aprobadas quedan registradas en el historial del empleado."),
    ("Importación CSV", "La importación masiva se realiza desde Administración, Importar empleados. El archivo debe ser CSV UTF-8 e incluir las columnas nombre, apellido, correo, país, fecha_ingreso y rol. Antes de confirmar, PeopleFlow muestra una vista previa y reporta filas inválidas. Las filas con un correo ya existente no crean un duplicado; se informan para revisión. La importación guarda un reporte descargable con filas creadas, actualizadas y rechazadas."),
    ("Reportes", "Los reportes de dotación, ausentismo y vacaciones se consultan desde Analítica. Administradores acceden a reportes globales y líderes solo ven su jerarquía. Un reporte puede exportarse como CSV cuando el usuario tiene permiso de exportación. Los datos se actualizan cada noche y la pantalla muestra la fecha de última actualización. Los filtros por país, área y período se aplican antes de descargar el archivo."),
    ("Notificaciones", "PeopleFlow envía notificaciones por correo y dentro de la aplicación. Cada usuario puede elegir qué recordatorios recibe, excepto mensajes obligatorios de seguridad y cumplimiento. Los administradores configuran plantillas de bienvenida, vencimiento de documentos y recordatorios de aprobación. Si un correo rebota, el sistema conserva el evento en auditoría y recomienda verificar el correo del usuario."),
    ("Integraciones", "Las integraciones disponibles se gestionan desde Configuración, Integraciones. Cada integración utiliza credenciales propias y un administrador debe autorizarla. PeopleFlow registra la fecha de última sincronización y los errores de cada ejecución. Una integración puede pausarse sin eliminar su configuración. Al desconectarla, el sistema deja de sincronizar datos nuevos pero conserva el historial de auditoría."),
    ("Seguridad y auditoría", "La auditoría registra creación de usuarios, cambios de rol, exportaciones, inicios de sesión y modificaciones de configuraciones. Solo administradores autorizados pueden consultar este registro. Los eventos incluyen fecha, usuario, acción y origen. La exportación de datos personales debe respetar permisos y políticas internas. Ante un incidente, soporte debe revisar auditoría antes de modificar permisos o restaurar una cuenta."),
]
# Repetir cada sección con una aclaración operativa para garantizar un corpus suficientemente largo.
def main() -> None:
    settings = Settings.from_env()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    paragraphs = []
    for title, body in SECTIONS:
        paragraphs.extend([f"## {title}", body, "Esta regla se aplica de forma consistente y cualquier excepción debe quedar documentada por RR.HH. o un administrador autorizado."])
    document = "\n\n".join(paragraphs * 3)
    (settings.data_dir / "faq_document.txt").write_text(document, encoding="utf-8")
    golden = [
        {"question": "¿Cómo restablezco mi contraseña?", "expected_keywords": ["contraseña", "enlace"]},
        {"question": "¿Quién puede exportar un reporte?", "expected_keywords": ["exportar", "permiso"]},
        {"question": "¿Qué pasa con una licencia sin comprobante?", "expected_keywords": ["comprobante", "pendiente"]},
    ]
    (settings.data_dir / "golden_cases.json").write_text(
        json.dumps(golden, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Datos generados.")


if __name__ == "__main__":
    main()

