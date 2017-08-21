/** \file profiler.h

    File imported from "common" lib, use if this library is not available.

    Implements basic profiling code.
    Access global profiler object using a call to profiler().

    Copyright (C) 2001 Bastian Goldluecke,
    <first name>AT<last name>.net

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

#ifndef __COCO_PROFILER_H
#define __COCO_PROFILER_H

#include <time.h>
#include <map>

#include "debug.h"



namespace coco
{
  /// Class for profiling applications.
  /** Several tasks can be profiled independently and are identified by a name.
      In order to profile, create an object of this class, and wrap the profiled
      code in beginTask() and endTask() calls. The results are printed to the debug
      stream on demand, or on destruction of the object.
      \ingroup sysprofile
   */
  class Profiler
  {
  public:

    /// Constructor.
    Profiler();
    /// Destructor. Results are print out when benchmark terminates.
    ~Profiler();

    /// Tell the profiler that a task has started
    void beginTask( const std::string &task );
    /// Tell the profiler that a task has ended
    void endTask( const std::string &task );

    /// Return the time used for the last run of the task
    clock_t timeLastRun( const std::string &task ) const;
    /// Total time used for the task so far
    clock_t timeTotal( const std::string &task ) const;

    /// Profile table for all tasks
    void printProfile() const;
    void printProfile( std::ostream &o ) const;

  private:

    /// Private info structure for benchmark class. Internal use only.
    struct S_Info {
      clock_t total;
      int     count;
      S_Info() {total=0;count=0;};
    };

    /// Starting times for tasks
    std::map<std::string, clock_t> _startTime;
    /// Time used for last run
    std::map<std::string, clock_t> _lastTime;
    /// Total time used
    std::map<std::string, S_Info> _totalTime;
  };


  /// Global Profiling object
  Profiler *profiler();


}


#endif
