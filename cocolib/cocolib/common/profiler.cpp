/** \file profiler.cpp

    File imported from "common" lib, use if this library is not available.

    Implements basic profiling code.
    Access global profiler object using a call to profiler().

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
   
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "profiler.h"

namespace coco {

  using namespace std;


  /// Constructor.
  Profiler::Profiler()
  {
  }

  /// Destructor. Results are print out when benchmark terminates.
  Profiler::~Profiler()
  {
  }

  
  /// Indicates that a task has started
  void Profiler::beginTask( const string &task )
  {
    _startTime[task] = clock();
  }

    /// Indicates that a task has ended
  void Profiler::endTask( const string &task )
  {
    clock_t tm = clock();
    map<string,clock_t>::const_iterator it = _startTime.find(task);
    if (it != _startTime.end()) {
      map<string,S_Info>::iterator itt = _totalTime.find(task);
      tm -= (*it).second;
      if (itt != _totalTime.end()) {
        (*itt).second.total += tm;
	(*itt).second.count++;
      }
      else {
	_totalTime[task].total = tm;
	_totalTime[task].count = 0;
      }
      _lastTime[task] = tm;
    }
    else {
      ERROR( "Profiler: ended undefined task " << task << endl );
    }
  }

  /// Time used for the last run of the task
  clock_t Profiler::timeLastRun( const string &task ) const
  {
    map<string,clock_t>::const_iterator it = _lastTime.find(task);
    if (it != _lastTime.end()) {
      return (*it).second;
    }

    ERROR( "Profiler: undefined task " << task << endl );
    return 0;
  }

  /// Total time used for the task so far
  clock_t Profiler::timeTotal( const string &task ) const
  {
    map<string,S_Info>::const_iterator it = _totalTime.find(task);
    if (it != _totalTime.end()) {
      return (*it).second.total;
    }

    ERROR( "Profiler: undefined task " << task << endl );
    return 0;
  }

  /// Profile table for all tasks
  void Profiler::printProfile( std::ostream &o ) const
  {
    char str[200];
    sprintf( str, "%20s Avg  Tot   Percentage", "Task" );
    o << str << endl;

    float ftotal = 0.0f;
    map<string,S_Info>::const_iterator it = _totalTime.begin();
    while (it != _totalTime.end()) {
      float fsecs = float( (*it).second.total ) / CLOCKS_PER_SEC;
      ftotal += fsecs;
      it++;
    }

    if (ftotal==0.0) ftotal=1.0f;
    
    it = _totalTime.begin();
    while (it != _totalTime.end()) {
      float fsecs = float( (*it).second.total ) / CLOCKS_PER_SEC;
      float fcount = float( (*it).second.count + 1);
      sprintf( str, "%20s %3.2fs %3.2fs %3.2f", (*it).first.c_str(), fsecs/fcount, fsecs, fsecs/ftotal );
      o << str << endl;
      it++;
    }
  }

  /// Profile table for all tasks
  void Profiler::printProfile() const
  {
    printProfile( debugStream() );
  }

  /// Global Profiling object
  Profiler __profiler;
  Profiler *profiler() {
	return &__profiler;
  }
}
