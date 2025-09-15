from agentix.stack import ViewRouter
from .views.index import IndexView
from .views.property_list import PropertyListView
from .views.property_edit import PropertyEditView
from .views.property_create import PropertyCreateView
from .views.property_delete import PropertyDeleteView
from .views.client_select import ClientSelectView

def build_router() -> ViewRouter:
    router = ViewRouter()
    router.register(IndexView.screen_key, lambda: IndexView())
    router.register(PropertyListView.screen_key, lambda: PropertyListView())
    router.register(PropertyEditView.screen_key, lambda: PropertyEditView())
    router.register(PropertyCreateView.screen_key, lambda: PropertyCreateView())
    router.register(PropertyDeleteView.screen_key, lambda: PropertyDeleteView())
    router.register(ClientSelectView.screen_key, lambda: ClientSelectView())
    router.set_index(IndexView.screen_key)
    return router
