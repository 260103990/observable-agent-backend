import asyncio

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings


router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get('/ready', responses={503: {'description': 'Local initialization or database check failed'}},
            description='Checks local startup, configured database connectivity, and enabled RAG initialization. '
                        'Does not check database schema, remote model APIs, or answer quality.')
async def readiness(request: Request, settings: Settings = Depends(get_settings)) -> JSONResponse:
    checks = {'startup': 'incomplete', 'database': 'not_checked', 'rag': 'not_checked'}
    ready = False
    if getattr(request.app.state, 'startup_complete', False):
        checks['startup'] = 'complete'
        checks['rag'] = ('initialized' if getattr(request.app.state, 'rag_service', None) is not None
                         else 'unavailable') if settings.rag_enabled else 'disabled'
        checks['database'] = 'not_configured'
        if settings.database_url is not None:
            checks['database'] = 'unavailable'
            pool = getattr(request.app.state, 'db_pool', None)
            if pool is not None:
                budget = settings.readiness_timeout_seconds
                try:
                    # Acquisition + query share a deadline. asyncpg's release
                    # may need its own bounded cleanup after cancellation.
                    async with asyncio.timeout(budget):
                        async with pool.acquire(timeout=budget) as connection:
                            if await connection.fetchval('SELECT 1', timeout=budget) == 1:
                                checks['database'] = 'ok'
                except Exception:
                    # Includes release failure; never report success before
                    # the connection has been returned. Cancellation propagates.
                    checks['database'] = 'unavailable'
        ready = checks['database'] in ('ok', 'not_configured') and checks['rag'] != 'unavailable'
    return JSONResponse({'status': 'ready' if ready else 'not_ready', 'checks': checks},
                        status_code=200 if ready else 503, headers={'Cache-Control': 'no-store'})
