from pathlib import Path
root=Path(__file__).parent
(root/"faq.txt").write_text("PeopleFlow permite recuperar contraseñas desde el inicio de sesión. El enlace se envía al correo laboral y vence en treinta minutos. Las vacaciones se solicitan desde Mi tiempo y requieren aprobación del líder. Las licencias médicas requieren documentación y RR.HH. valida los comprobantes. Los administradores pueden crear usuarios, definir roles y consultar auditoría. Los reportes se exportan solo si el rol tiene permiso. La plataforma registra integraciones y sincronizaciones. Cuando el contexto no contiene una respuesta, el asistente debe decir que no hay información suficiente.",encoding="utf-8")
print("Datos generados.")

