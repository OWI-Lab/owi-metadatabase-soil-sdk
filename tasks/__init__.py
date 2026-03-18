from invoke.collection import Collection

from . import docs, performance, quality, search, test

ns = Collection()
ns.add_collection(quality, name="qa")
ns.add_collection(docs)
ns.add_collection(test)
ns.add_collection(search)
ns.add_collection(performance)
