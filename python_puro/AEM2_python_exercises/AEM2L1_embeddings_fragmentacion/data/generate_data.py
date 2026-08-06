"""Regenera datos locales para AEM2L1_embeddings_fragmentacion."""
from pathlib import Path
root = Path(__file__).parent
# El corpus es pequeño y local para que el ejercicio se pueda regenerar siempre igual.
(root / "policies.txt").write_text("Las políticas de acceso indican que la contraseña se recupera mediante un enlace enviado al correo laboral. El enlace vence y soporte valida la identidad antes de desbloquear una cuenta. Las políticas de vacaciones exigen una aprobación del líder y conservan el historial. La conectividad de la oficina se informa desde soporte y las incidencias se registran con fecha, usuario y prioridad. Los documentos de seguridad describen autenticación, roles, permisos y auditoría.", encoding="utf-8")
print("Datos regenerados.")
