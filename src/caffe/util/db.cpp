#include "caffe/util/db.hpp"
#include "caffe/util/db_leveldb.hpp"
#include "caffe/util/db_lmdb.hpp"

#include <string>

namespace caffe { namespace db {

DB* GetDB(DataParameter::DB backend) {
  switch (backend) {
  case DataParameter_DB_LEVELDB:
    return new LevelDB();
#ifdef LMDB
  case DataParameter_DB_LMDB:
    return new LMDB();
#endif // LMDB
  default:
    LOG(FATAL) << "Unknown database backend";
  }
}

DB* GetDB(const string& backend) {
  if (backend == "leveldb") {
    return new LevelDB();
#ifdef LMDB
  } else if (backend == "lmdb") {
    return new LMDB();
#endif // LMDB
  } else {
    LOG(FATAL) << "Unknown database backend";
  }
}

}  // namespace db
}  // namespace caffe
